# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "dpo-qwen"  # optional: name your project
os.environ["WANDB_NAME"] = "dpo-qwen-2.5-0.5b-instruct-run-5"  # set the specific run name

import torch
import random
from threading import Thread
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer, TextStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOConfig, DPOTrainer

# # #  ----------- Load Model and Tokenizer ----------- # # #

model_path = "/nfs1/jiaxinzhang/models/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fase=True, padding_side="left")
# tokenizer = AutoTokenizer.from_pretrained("/home/jiaxijzhang/llm_relevant_study/rl/dpo/saved_models/dpo-qwen-2.5-0.5b-instruct-run-5-DPO-bad-boy/merged_model", use_fase=True)

# # #  ----------- Create Dataset ----------- # # #

dataset = load_dataset("/home/jiaxijzhang/llm_relevant_study/rl/dpo/btfChinese-DPO-small")
train_data = dataset["train"]

def qwen_format_question(question, tokenize_or_not=False):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
            {"role": "user", "content": question}
        ],
        tokenize=False,
        add_generation_prompt=True,
    ) if not tokenize_or_not else tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
            {"role": "user", "content": question}
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",   
    )

formatted_data = [
    {
        "prompt": qwen_format_question(data["question"]),
        "chosen": data["chosen"],
        "rejected": data["rejected"],
    }

    for data in train_data
]
all_indices = list(range(len(formatted_data)))
random.shuffle(all_indices)

split_index = int(len(formatted_data) * 0.8)
train_indices = all_indices[: split_index]
test_indices = all_indices[split_index :]

reformatted_dataset = {
    "train": [formatted_data[i] for i in train_indices],
    "test": [formatted_data[i] for i in test_indices],
}

train_dataset = Dataset.from_list(reformatted_dataset["train"])
test_dataset = Dataset.from_list(reformatted_dataset["test"])

# # #  ----------- Generate Function ----------- # # #

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_answer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generate_prompt=True, return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids,
        max_length=1024,
        temperature=0.7,
        top_p=0.9,
        stopping_criteria=[StopOnTokens([tokenizer.eos_token_id])]
    )
    return tokenizer.decode(outputs[0], skip_sepcial_tokens=True)

# prompt = "你是谁"
# generated_text = generate_answer(model, tokenizer,  prompt)
# print(generated_text)

model.gradient_checkpointing_enable()
def print_trainable_parameters(model):
    trainable_params = 0
    non_trainable_params = 0
    all_params = 0

    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f" {name}")
        else:
            non_trainable_params += param.numel()
    
    print("---")
    print("Non-Trainable Parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f" {name}")
    
    print("---")
    print(f"Trainable parameters: {trainable_params}\n Non-Trainable: {non_trainable_params}\n Trainable: {100 * trainable_params / all_params:.2f}%")

# # #  ----------- LoRA Config ----------- # # #

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        "self_attn.q_proj", # Self-attention的Query投影
        "self_attn.k_proj", # Self-attention的Key投影  
        "self_attn.v_proj", # Self-attention的Value投影
        "self_attn.o_proj", # Self-attention的输出投影
        # "self_attn.rotary_emb.inv_freq", # 旋转位置编码,一般不需要微调
        "mlp.gate_proj", # MLP门控投影
        "mlp.up_proj", # MLP上投影
        "mlp.down_proj", # MLP下投影
        # "input_layernorm.weight",  # 输入归一化层
        # "post_attention_layernorm.weight", # Attention后面的LayerNorm层
        # "model.norm.weight", # 模型归一化层
        # "lm_head.weight", # 语言模型输出层
        # "dense_h_to_4h", # Falcon模型特有的全连接层
        # "dense_4h_to_h", # Falcon模型特有的全连接层
        # "query_key_value", # Falcon模型的QKV合并层
        # "dense" # Falcon模型特有的全连接层
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# model = get_peft_model(model, peft_config)
# print_trainable_parameters(model)

if '<pad>' not in tokenizer.get_vocab():
    added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
else:
    added_tokens = 0

# 检查模型是否需要调整大小
if added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))
    print('Resizing token embeddings！')

# 在模型中配置填充标记
model.config.pad_token_id = tokenizer.pad_token_id

assert model.config.pad_token_id == tokenizer.pad_token_id, "模型的填充标记ID与分词器的填充标记ID不匹配！"
assert model.config.eos_token_id == tokenizer.eos_token_id, "模型的结束标记ID与分词器的结束标记ID不匹配！"

# 更新分词器的最大长度以匹配模型配置的最大positional embedding
tokenizer.model_max_length = model.config.max_position_embeddings
tokenizer.padding_side  = 'left'
print("Tokenizer vocab_size:", tokenizer.vocab_size)
print("Special tokens map:", tokenizer.special_tokens_map)

# # #  ----------- Evaluation ----------- # # #

def stream(user_prompt, model):
    
    input_ids = qwen_format_question(question=user_prompt, tokenize_or_not=True).to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    generation_kwargs = {
        "inputs": input_ids,
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
        "stopping_criteria": [StopOnTokens([tokenizer.eos_token_id])]
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 实时输出生成的文本
    generated_text = ""
    # 获取输入的长度
    token_counts = 0
    for new_text in streamer:
        if token_counts < 4:
            token_counts += 1
            continue
        print(new_text, end="", flush=True)
        generated_text += new_text

    torch.cuda.empty_cache()

def evaluation(model_type, base_model, questions, adapter_checkpoint="", save_model=""):
    if model_type == "base":
        eval_model = base_model
    elif model_type == "fine-tuned":
        eval_model = PeftModel.from_pretrained(base_model, adapter_checkpoint)
        eval_model = eval_model.merge_and_unload()
        eval_model.save_pretrained(save_model, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(save_model)

    eval_model = eval_model.to("cuda")

    for que in questions:
        stream(que, eval_model)
        print("\n")

# evaluation("base")

# # #  ----------- Train ----------- # # #

context_length = 512 * 4
grad_accum = 2
batch_size = 4
fine_tune_tag = "DPO-bad-boy"

epochs = 3
save_dir = f"./saved_models/{os.environ["WANDB_NAME"]}-{fine_tune_tag}"

training_arguments = DPOConfig(
    output_dir=save_dir,
    eval_strategy="steps",
    beta=0.1,
    do_eval=True,
    eval_steps=0.25,
    optim="adamw_torch",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    per_device_eval_batch_size=batch_size,
    log_level="debug",
    save_steps=0.25,
    logging_steps=1,
    bf16=True,     
    learning_rate=1e-6,
    num_train_epochs=epochs,
    lr_scheduler_type="linear",
    report_to="wandb",
)

trainer = DPOTrainer(
    model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
)
model.config.use_cache = False
trainer.train()

model = AutoModelForCausalLM.from_pretrained(
    "/home/jiaxijzhang/llm_relevant_study/rl/dpo/saved_models/dpo-qwen-2.5-0.5b-instruct-run-5-DPO-bad-boy/merged_model",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = True

questions = [
    "程序员的悲哀是什么？",
    "告诉我，数据科学家是科学家吗？",
    "为什么年轻人不买房了？",
    "如何评价中医？",
    "怎么理解“真传一句话，假传万卷书”？"
]


evaluation("base", model, questions)
# evaluation("fine-tuned", model, questions, "/home/jiaxijzhang/llm_relevant_study/rl/dpo/saved_models/dpo-qwen-2.5-0.5b-instruct-run-5-DPO-bad-boy/checkpoint-1500", 
#     "/home/jiaxijzhang/llm_relevant_study/rl/dpo/saved_models/dpo-qwen-2.5-0.5b-instruct-run-5-DPO-bad-boy/merged_model")