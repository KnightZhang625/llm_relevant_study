# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("/home/jiaxijzhang/llm_relevant_study/rl/grpo/runs/qwen-2.5-3b-r1-countdown/checkpoint-450")

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
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

dataset_path = "/home/jiaxijzhang/llm_relevant_study/rl/grpo/datasets/Qwen2.5-3B-Instruct-Countdown-Tasks-3to4"
dataset = load_dataset(dataset_path, split="train")

input_text = generate_r1_prompt(dataset[0]["nums"], dataset[0]["target"])["prompt"]

llm = LLM(
    model="/home/jiaxijzhang/llm_relevant_study/rl/grpo/runs/qwen-2.5-3b-r1-countdown",
    tokenizer="/home/jiaxijzhang/llm_relevant_study/rl/grpo/runs/qwen-2.5-3b-r1-countdown",
    dtype="bfloat16",
    max_num_seqs=512,
    swap_space=16,
)

sampling_params = SamplingParams(
    temperature=0,
    top_p=1,
    top_k=-1,
    repetition_penalty=1.1,
    n=1,
    max_tokens=1024,
    logprobs=1,
    stop=[],
    best_of=1,
)
print(input_text)
output = llm.generate(input_text, sampling_params=sampling_params, use_tqdm=False)
print([[o.text for o in resp_to_single_input.outputs] for resp_to_single_input in output])