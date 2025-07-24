# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from threading import Thread

def convert_data_to_dpo_format(query: str, tokenizer: AutoTokenizer=None, return_tensors: bool=False):
    query_message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    tokens = tokenizer.apply_chat_template(
        query_message,
        tokenize=False,
        add_generation_format=True,
    )
    return tokenizer(tokens, add_special_tokens=False, return_tensors="pt")

def generate_answer_non_stream(query: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):

    inputs = convert_data_to_dpo_format(query, tokenizer, return_tensors=True)

    generation_kwargs = {
        "input_ids": inputs["input_ids"].to(model.device),
        "attention_mask": inputs["attention_mask"].to(model.device),
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }

    output = model.generate(
        **generation_kwargs
    )

    return tokenizer.decode(output[0][inputs["input_ids"].size(1):], skip_special_tokens=True)

def generate_answer_stream(query: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    inputs = convert_data_to_dpo_format(query, tokenizer=tokenizer, return_tensors=True)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    generation_kwargs = {
        "input_ids": inputs["input_ids"].to(model.device),
        "attention_mask": inputs["attention_mask"].to(model.device),
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
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
    )
    print(f"Use key value cache? : {model.config.use_cache}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    predictions = []
    for query in queries:
        ans = generate_answer_non_stream(query, model, tokenizer)
        if verbose:
            print(f"Que: {query}\nAns: {ans}\n")

def evaluation_streamer(model_path: str):
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
    model_path = "/nfs1/jiaxinzhang/saved_for_checkpoint/merged-dpo-llama-3-1-8b-math-ep3"

    queries = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    ]

    # evaluation(queries, model_path, None, True)

    evaluation_streamer(model_path)