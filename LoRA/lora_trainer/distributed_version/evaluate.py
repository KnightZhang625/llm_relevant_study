# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
from peft import PeftModel

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: list[int]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, score, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def convert_to_qwen_chat_format(query: str, tokenizer: AutoTokenizer):
    message = [
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": query},
    ]

    tokens = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(tokens, add_special_tokens=False, return_tensors="pt")

    return inputs

def generate_answer_for_query(query: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    inputs = convert_to_qwen_chat_format(query, tokenizer)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    generation_kwargs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
        "stopping_criteria": [StopOnTokens([tokenizer.eos_token_id])],
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Que: ", query)
    print("Ans: ", end="")
    for new_token in streamer:
        print(new_token, end="", flush=True)

def evaluation(queries: list[str], model_path: str, adapter_path: str=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2"
    ).to(device)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        ).to(device)
        model.merge_and_unload()


    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for query in queries:
        generate_answer_for_query(query, model, tokenizer)
        print("\n\n\n")

queries = [
    "程序员的悲哀是什么？",
    "告诉我，数据科学家是科学家吗？",
    "为什么年轻人不买房了？",
    "如何评价中医？",
    "怎么理解“真传一句话，假传万卷书”？"
]
model_path = "/nfs1/jiaxinzhang/models/Qwen2.5-0.5B-Instruct"
adapter_path = "/home/jiaxijzhang/llm_relevant_study/LoRA/lora_trainer/saved_models"

evaluation(queries, model_path, adapter_path)