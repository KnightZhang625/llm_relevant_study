# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
from datetime import datetime
from threading import Thread
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
from utils import process_dataset, convert_data_to_qwen_format

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

def generate_answer_non_stream(query: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):

    inputs = convert_data_to_qwen_format(query, chosen=None, rejected=None, tokenizer=tokenizer, return_tensors=True)

    generation_kwargs = {
        "input_ids": inputs["input_ids"].to(model.device),
        "attention_mask": inputs["attention_mask"].to(model.device),
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "stopping_criteria": [StopOnTokens([tokenizer.eos_token_id])]
    }

    output = model.generate(
        **generation_kwargs
    )

    return tokenizer.decode(output[0][inputs["input_ids"].size(1):], skip_special_tokens=True)

def generate_answer_stream(query: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    inputs = convert_data_to_qwen_format(query, chosen=None, rejected=None, tokenizer=tokenizer, return_tensors=True)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    generation_kwargs = {
        "input_ids": inputs["input_ids"].to(model.device),
        "attention_mask": inputs["attention_mask"].to(model.device),
        "max_length": 512,
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

def evaluation(queries: list[str], model_path: str, save_dir: str, verbose: bool=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    predictions = []
    for query in queries:
        ans = generate_answer_non_stream(query, model, tokenizer)
        if verbose:
            print(f"Que: {query}\nAns: {ans}\n")
        predictions.append({"Que": query, "Ans": ans})
    
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path(save_dir).mkdir(exist_ok=True)
    with open(Path(save_dir) / f"predictions_{date}.json", "w", encoding="utf-8") as file:
        for pred in predictions:
            file.write(
                json.dumps(pred, ensure_ascii=False, indent=2) + "\n"
            )

def evaluatio_streamer(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = ""
    c = 0
    while True:
        prompt = input("Type your prompt: ")

        if prompt == "eos":
            break

        generate_answer_stream(prompt, model, tokenizer)
        c +=1
        print(c)
        print("\n", "*"*100, "\n")

    print("Bye.")

if __name__ == "__main__":
    queries = [
        "程序员的悲哀是什么？",
        "告诉我，数据科学家是科学家吗？",
        "为什么年轻人不买房了？",
        "如何评价中医？",
        "怎么理解“真传一句话，假传万卷书”？"
    ]
    evaluation(queries, "/home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_distributed/saved_models", f"pred_resutls", verbose=True)
    # evaluatio_streamer("/home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_distributed/saved_models")