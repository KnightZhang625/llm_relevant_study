# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "sft-lora-qwen"  # optional: name your project
# os.environ["WANDB_NAME"] = "sft-lora-qwen-2.5-1.5b-instruct-run-1"  # set the specific run name

import torch
from datasets import load_dataset, Dataset
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format
from peft import LoraConfig, PeftModel

def process_dataset(data):
    """Accepted dataset format 1., in conversation format, with 'messages' as key."""
    message = [
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": data["question"]},
        {"role": "assistant", "content": data["chosen"]},
    ]
    return {"messages": message}

def tokenize_dataset(data: dict[str, str], tokenizer: AutoTokenizer):
    """Accepted dataset format 2., in conversation format, tokenized, then use 'text' as key."""
    message = [
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": data["question"]},
        {"role": "assistant", "content": data["chosen"]},
    ]
    # NOTE: if we apply chat template, the name must be "text" in SFT.
    return {"text" : tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )}

dataset = load_dataset("/home/jiaxijzhang/llm_relevant_study/rl/dpo/datasets/btfChinese-DPO-small", split="train")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_path = "/nfs1/jiaxinzhang/models/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_path)

# NOTE: Format 1: just conversation format
# dataset = dataset.map(lambda x: process_dataset(x), remove_columns=["system", "question", "chosen", "rejected"])

# NOTE: Foramt 2: we apply chat template first.
tokenize_dataset_fn = partial(tokenize_dataset, tokenizer=tokenizer)
dataset = dataset.map(lambda x: tokenize_dataset_fn(x), remove_columns=['system', 'question', 'chosen', 'rejected'])
dataset = Dataset.from_list(dataset)

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="saved_models",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=1e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.3,
    lr_scheduler_type="constant",
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    max_length=1024,
    packing=True,
    report_to="wandb",
    run_name="sft-lora-qwen-2.5-1.5b-instruct-run-1",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model()


# merged_model = PeftModel.from_pretrained(
#     model,
#     "/home/jiaxijzhang/llm_relevant_study/LoRA/lora_trainer/distributed_version/saved_models"
# ).to(device)
# merged_model = merged_model.merge_and_unload()

# queries = [
#     "程序员的悲哀是什么？",
#     "告诉我，数据科学家是科学家吗？",
#     "为什么年轻人不买房了？",
#     "如何评价中医？",
#     "怎么理解“真传一句话，假传万卷书”？",
#     "上海和北京怎么选？"
# ]

# for query in queries:
#     message = [
#         {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
#         {"role": "user", "content": query},
#     ]
#     tokens = tokenizer.apply_chat_template(
#         message,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     inputs = tokenizer(tokens, add_special_tokens=False, return_tensors="pt")

#     output = merged_model.generate(
#         input_ids=inputs["input_ids"].to(device),
#         attention_mask=inputs["attention_mask"].to(device),
#         max_length=512,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     print(tokenizer.decode(output[0][inputs["input_ids"].size(1):], skip_special_tokens=True))