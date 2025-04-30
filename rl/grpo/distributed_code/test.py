# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from threading import Thread
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria

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

def generate_r1_prompt(numbers, target, tokenizer):
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

    return tokenizer.apply_chat_template(r1_prefix, tokenize=True, continue_final_message=True, return_tensors="pt")

def generate_answer_stream(query: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """Generate answer in the stream way."""

    input_ids = generate_r1_prompt(query["nums"], query["target"], tokenizer).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    generation_kwargs = {
        "input_ids": input_ids,
        "max_length": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
        "stopping_criteria": [StopOnTokens([tokenizer.eos_token_id])]
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Que: ", tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    print("Ans: ", end="")
    for new_token in streamer:
        print(new_token, end="", flush=True)

dataset_path = "/home/jiaxijzhang/llm_relevant_study/rl/grpo/datasets/Qwen2.5-3B-Instruct-Countdown-Tasks-3to4"
dataset = load_dataset(dataset_path, split="train")

model = AutoModelForCausalLM.from_pretrained(
        "/nfs1/jiaxinzhang/models/Qwen2.5-0.5B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
model.config.use_cache = True
tokenizer = AutoTokenizer.from_pretrained("/nfs1/jiaxinzhang/models/Qwen2.5-0.5B-Instruct")

generate_answer_stream(
    dataset[0],
    model,
    tokenizer,
)