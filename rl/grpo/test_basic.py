# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
        "/home/jiaxijzhang/llm_relevant_study/rl/grpo/distributed_code/saved_models/qwen-2.5-3b-r1-countdown",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
model.config.use_cache = True
tokenizer = AutoTokenizer.from_pretrained("/home/jiaxijzhang/llm_relevant_study/rl/grpo/distributed_code/saved_models/qwen-2.5-3b-r1-countdown")

def generate_r1_prompt(numbers, target):
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        {
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=True, continue_final_message=True, return_tensors="pt"), "target": target}

dataset_path = "/home/jiaxijzhang/llm_relevant_study/rl/grpo/datasets/Qwen2.5-3B-Instruct-Countdown-Tasks-3to4"
dataset = load_dataset(dataset_path, split="train")

input_ids = generate_r1_prompt(dataset[0]["nums"], dataset[0]["target"])["prompt"].to(model.device)

output = model.generate(
    input_ids,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))