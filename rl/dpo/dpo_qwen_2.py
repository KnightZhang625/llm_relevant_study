# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "dpo-qwen"  # optional: name your project
os.environ["WANDB_NAME"] = "dpo-qwen-2.5-0.5b-instruct-run-8"  # set the specific run name

import torch
import random
from pathlib import Path
from threading import Thread
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOConfig, DPOTrainer

@dataclass
class Config:

    # Data
    raw_data_path: str

    # Model
    base_model_path: str
    adapter_path: str = ""

    # Training
    enable_lora: bool = True
    context_length: int = 512 * 4
    grad_accu: int = 2
    batch_size: int = 16
    epoch: int = 3
    lr: float = 1e-5

    save_dir_name: str = "dpo_qwen_saved_models"
    merge_dir_name: str = "merged_model"

def convert_data_to_qwen_format(query: str, tokenizer: AutoTokenizer, to_tokenize: bool=False):
    """Convert query (plain str) to the chat_template format."""

    message = [
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": query}
    ]
    if not to_tokenize:
        return tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # we can use "tokenize=True, return_tensors='pt' ", however, this will only return the input_ids of the message, lacking "attention_mask",
        # will cause the alert: "The attention mask is not set and cannot be inferred from input because pad token is same as eos token. 
        #   As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
        # Therefore, we use tokenizer to get attention_mask.
        tokens = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(tokens, add_special_tokens=False, return_tensors="pt")


def process_dataset(raw_dataset: Dataset, tokenizer: AutoTokenizer, shuffle: bool=True, split_ration: float=0.8) -> tuple[Dataset, Dataset]:
    """Convert each data from the raw dataset into qwen chat_template."""

    formatted_dataset = [
        {
            "prompt": convert_data_to_qwen_format(query=data["question"], tokenizer=tokenizer, to_tokenize=False),
            "chosen": data["chosen"],
            "rejected": data["rejected"],
        }
        for data in raw_dataset
    ]

    all_indices = list(range(len(formatted_dataset)))
    if shuffle:
        random.shuffle(all_indices)
    split_index = int(len(formatted_dataset) * split_ration)
    train_indices = all_indices[:split_index]
    test_indices = all_indices[split_index:]

    formatted_dataset = {
        "train": [formatted_dataset[i] for i in train_indices],
        "test": [formatted_dataset[i] for i in test_indices],
    }

    train_dataset = Dataset.from_list(formatted_dataset["train"])
    test_dataset = Dataset.from_list(formatted_dataset["test"])

    return train_dataset, test_dataset

class StopOnTokens(StoppingCriteria):
    """
        Good for streaming generation (TextStreamer) because you can stop token-by-token dynamically.
        Because eos_token_id is a generation finalization argument, not a real-time stopping signal.
        Think about it:
        During normal generation Huggingface first generates the full sequence, then checks:
        "Hey, did any sequence hit eos_token_id? If yes, truncate."
        BUT during streaming, there is no time to check that because tokens are sent immediately to you!
    """
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
    
    def __call__(self, input_ids, score, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_answer_non_stream(query: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, system_prompt: str=None):
    """Generate answer in the non-stream way."""

    message = []
    if system_prompt is not None:
        message.append({"role": "system", "content": system_prompt})
    message.append({"role": "user", "content": query})

    input_ids = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        input_ids,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_answer_stream(query: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """Generate answer in the stream way."""

    inputs = convert_data_to_qwen_format(query, tokenizer=tokenizer, to_tokenize=True)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    generation_kwargs = {
        "input_ids": inputs["input_ids"].to(model.device),
        "attention_mask": inputs["attention_mask"].to(model.device),
        "max_length": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
        "stopping_criteria": [StopOnTokens([tokenizer.eos_token_id])]
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Que: ", query)
    print("Ans: ", end="")
    for new_token in streamer:
        print(new_token, end="", flush=True)

def print_trainable_parameters(model):
    trainable_params = 0
    non_trainable_params = 0
    all_params = 0

    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            non_trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params}\n Non-Trainable: {non_trainable_params}\n Trainable: {100 * trainable_params / all_params:.2f}%")

def train(config: Config):
    # ----- load model, tokenizer ----- #
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, use_fase=True, padding_side="left")

    try:
        print("*"*100, "\n", "Testing llm...")
        print(generate_answer_non_stream("你是谁", base_model, tokenizer), "\n")
    except Exception as e:
        print("Test failed.")
        raise e

    # ----- load dataset ----- #
    raw_dataset = load_dataset(config.raw_data_path)["train"]
    train_dataset, test_dataset = process_dataset(raw_dataset, tokenizer)
    
    # ----- Lora Config ----- #
    if config.enable_lora:
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
        peft_model = get_peft_model(base_model, peft_config)
        print_trainable_parameters(peft_model)
    
    # ----- Training ----- #
    training_arguments = DPOConfig(
        output_dir=Path(__file__).absolute().parent / config.save_dir_name,
        eval_strategy="steps",
        beta=0.1,   # Higher β means less deviation from the reference model
        do_eval=True,
        eval_steps=0.25,
        optim="adamw_torch",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accu,
        per_device_eval_batch_size=config.batch_size,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=1,
        bf16=True,
        learning_rate=config.lr,
        num_train_epochs=config.epoch,
        lr_scheduler_type="linear",
        report_to="wandb",
    )
    peft_model.config.use_cache = False
    trainer = DPOTrainer(
        peft_model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    torch.cuda.empty_cache()

def merge_lora_and_save(base_model_path: str, adapter_path: str, save_path: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(save_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(save_path, safe_serialization=True, max_shard_size="2GB")

    print(f"Model and Tokenizer are saved into: {save_path}")

def evaluation(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, queries: list[str]):
    model = model.to("cuda")

    for q in queries:
        generate_answer_stream(q, model, tokenizer)
        print("\n")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    config = Config(
        raw_data_path="/home/jiaxijzhang/llm_relevant_study/rl/dpo/btfChinese-DPO-small",
        base_model_path="/nfs1/jiaxinzhang/models/Qwen2.5-1.5B-Instruct",
        enable_lora=True
    )

    train(config)

    config.adapter_path = "/home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_qwen_saved_models/checkpoint-375"
    merge_lora_and_save(config.base_model_path, config.adapter_path, os.path.join(config.save_dir_name, config.merge_dir_name))

    queries = [
    "程序员的悲哀是什么？",
    "告诉我，数据科学家是科学家吗？",
    "为什么年轻人不买房了？",
    "如何评价中医？",
    "怎么理解“真传一句话，假传万卷书”？"
    ]   
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(config.save_dir_name, config.merge_dir_name),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.save_dir_name, config.merge_dir_name))
    evaluation(model, tokenizer, queries)