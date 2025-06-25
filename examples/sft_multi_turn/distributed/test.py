# coding:utf-8
# Author: Jiaxin
# Date: 17-June-2025

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

def convert_to_qwen_chat_format(query: list[str], tokenizer: AutoTokenizer):
    messages = [{"role": "system", "content": "你是一个指令和闲聊助手，请根据用户的输入判断用户输入是闲聊还是指令。"}]

    role = ["user", "assistant"]
    for i, que in enumerate(query):
        messages.append({"role": role[i % 2], "content": que})

    tokens = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    role = ["Que", "Ans"]
    for i, que in enumerate(query):
        print(f"{role[i % 2]}: {que}")
    print("Ans: ", end="")
    output_tokens = ""
    for new_token in streamer:
        output_tokens += new_token
        # print(new_token, end="", flush=True)
    return output_tokens

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

    # # multi-turns
    # for query in queries:
    #     ans = generate_answer_for_query(query, model, tokenizer)
    #     print(ans)
    #     print("\n\n\n")
    #     while True:
    #         query.append(ans)
    #         new_input = input("Your input: ")
    #         query.append(new_input)
    #         ans = generate_answer_for_query(query, model, tokenizer)
    #         print(ans)
    #         print("\n\n\n")
    
    # multi-turns
    inputs = []
    while True:
        query = input("Input: ")
        inputs.append(query)
        if query == "none":
            break
        ans = generate_answer_for_query(inputs, model, tokenizer)
        print(ans, "\n")
        inputs.append(ans)

    # # single-turn
    # while True:
    #     query = input("Input: ")
    #     if query == "none":
    #         break
    #     generate_answer_for_query([query], model, tokenizer)
    #     print("\n")

queries = [
    ["什么时候发货", "您好，我是小店【智能助理】。现在是客服休息时间，由我为您服务~", "什么时候发货", "亲亲~我们已经按照订单顺序陆续发出了哈，会尽快安排的，年节期间订单量多，请您再耐心等待哈", "我已经拼单成功14小时，能尽快帮我发货吗"]
]
model_path = "/nfs1/jiaxinzhang/saved_for_checkpoint/qwen2.5-7b-ins-chat-multi-round"
# model_path = "/nfs1/jiaxinzhang/saved_for_checkpoint/qwen2.5-7b-instruct-multi-turn"
adapter_path = None

evaluation(queries, model_path, adapter_path)