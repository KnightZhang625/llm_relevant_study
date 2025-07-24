import re
import json
import numpy as np
import concurrent.futures as futures
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from check_ins import check_ans

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
# openai_api_base = "http://11.215.122.101:56001/v1"
openai_api_base = "http://localhost:56001/v1"
# openai_api_base = "http://9.73.139.130:8100/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def request_api(messages: list[dict]):
    chat_response = client.chat.completions.create(
        model="qwen-7b",
        messages=messages,
        temperature=0.7,
        top_p=0.8,
        max_tokens=1024,
    )
    return chat_response.choices[0].message.content

def load_file(path):
    with open(path, "r", encoding="utf-8") as file:
        datas = json.load(file)
    if len(datas) > 3500:
        datas = np.random.choice(datas, size=3365, replace=False)
    return datas

def parse_instruct_gold_ans(ans) -> list:
    if "你说的是" not in ans and "关键信息是" not in ans:
        return ["空的", "空的"]
    matches = re.findall(r"你说的是 (.*?) 指令.*?其中[的]关键信息是 (.*?)。", ans)

    if matches:
        command, keyword = matches[0]
        if command == "": command = "空的" 
        if keyword == "": keyword = "空的"
        return [command, keyword]
    else:
        return ["空的", "空的"]

def parse_instruct_ans(ans) -> list:
    if "指令回复" not in ans:
        return ["空的", "空的"]
    
    matches = re.findall(r"行为\s*(.*?)\s*，关键词\s*(.*?)。", ans)

    if matches:
        command, keyword = matches[0]
        return [command, keyword]
    else:
        return ["空的", "空的"]


def test_multi_instruct(data):
    messages=[{"role": "system", "content": "你是一个指令和闲聊助手。"}]
    for his in data["history"]:
        messages.extend(
            [
                {"role": "user", "content": his[0]},
                {"role": "assistant", "content": his[1]},
            ]
        )
    messages.append({"role": "user", "content": data["data"]})

    ans = request_api(messages)
    parsed_res = parse_instruct_ans(ans)
    data["pred_intent"] = parsed_res[0]
    data["pred_item"] = parsed_res[1]
    data["output"] = ans

    return data

def test_multi_instruct_greedy(data):
    messages=[{"role": "system", "content": "你是一个指令和闲聊助手。"}]
    data["greedy_history"] = []
    for his in data["history"]:
        messages.append({"role": "user", "content": his[0]})
        ans = request_api(messages)
        messages.append({"role": "assistant", "content": ans})
        data["greedy_history"].append([his[0], ans])

    messages.append({"role": "user", "content": data["data"]})

    ans = request_api(messages)
    parsed_res = parse_instruct_ans(ans)
    data["pred_intent"] = parsed_res[0]
    data["pred_item"] = parsed_res[1]
    data["output"] = ans

    gold_parsed_res = parse_instruct_gold_ans(data["ref"])
    data["intent"] = gold_parsed_res[0]
    data["item"] = gold_parsed_res[1]

    return data

def test_multi_chat(data):
    messages=[{"role": "system", "content": "你你是一个指令和闲聊助手。"}]
    for his in data["history"]:
        messages.extend(
            [
                {"role": "user", "content": his[0]},
                {"role": "assistant", "content": his[1]},
            ]
        )
    messages.append({"role": "user", "content": data["data"]})
    ans = request_api(messages)
    data["output"] = ans

    return data

def test_multi_chat_greedy(data):
    messages=[{"role": "system", "content": data["system"]}]
    data["greedy_history"] = []
    for his in data["history"]:
        messages.append({"role": "user", "content": his[0]})
        ans = request_api(messages)
        messages.append({"role": "assistant", "content": ans})
        data["greedy_history"].append([his[0], ans])

    messages.append({"role": "user", "content": data["data"]})

    ans = request_api(messages)
    data["output"] = ans

    return data

def main(test_path, save_dir, is_instruction, test_greedy=False):
    cur_time = datetime.now().strftime("%d_%b")
    test_datas = load_file(test_path)

    if not test_greedy:
        test_fn = test_multi_instruct if is_instruction else test_multi_chat
        save_path = "multi_ins_predictions_with_oracle_7_July.json" if is_instruction else "multi_chat_predictions_with_oracle_7_July.json"
    else:
        # NOTE: 09-July: usually test greedy.
        test_fn = test_multi_instruct_greedy if is_instruction else test_multi_chat_greedy
        save_path = f"multi_ins_predictions_greedy_with_oracle_{cur_time}.json" if is_instruction else f"multi_chat_predictions_greedy_with_oracle_{cur_time}.json"

    threads = []
    predictions = []
    with futures.ThreadPoolExecutor(max_workers=16) as executor:
        for data in test_datas:
            threads.append(executor.submit(test_fn, data))

        for future in tqdm(futures.as_completed(threads), total=len(threads)):
            predictions.append(future.result())

    Path(save_dir).mkdir(exist_ok=True)
   
    with open(Path(save_dir) / save_path, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2, ensure_ascii=False)
    
    return Path(save_dir) / save_path

def eval_ins(path):
    predictions = []
    with open(path, "r", encoding="utf-8") as file:
        predictions = json.load(file)
    
    scores = []
    wrong_predictions = []
    for pred in predictions:
        if check_ans(pred, True):
            scores.append(1)
        else:
            scores.append(0)
            wrong_predictions.append(pred)
   
    return sum(scores) / len(scores), wrong_predictions

if __name__ == "__main__":
    # main("/home/jiaxijzhang/llm_relevant_study/dataset/test_multi_data/ins_multi_with_oracle.json", "/home/jiaxijzhang/llm_relevant_study/dataset/predictions_multi_data", True)
    # main("/home/jiaxijzhang/llm_relevant_study/dataset/test_multi_data/chat_multi_with_oracle.json", "/home/jiaxijzhang/llm_relevant_study/dataset/predictions_multi_data", False)

    save_path = main("/home/jiaxijzhang/llm_relevant_study/dataset/test_multi_data/test_7_July.json", "/home/jiaxijzhang/llm_relevant_study/dataset/predictions_multi_data", True, True)
    # main("/home/jiaxijzhang/llm_relevant_study/dataset/test_multi_data/chat_multi_4_July.json", "/home/jiaxijzhang/llm_relevant_study/dataset/predictions_multi_data", False, True)

    accu, wrong_predictions = eval_ins(save_path)
    print(f"Accu: {accu}")
    with open(Path("/home/jiaxijzhang/llm_relevant_study/dataset/predictions_multi_data") / "pred_ins_with_oracle_wrong_09_July.json", "w", encoding="utf-8") as file:
        json.dump(wrong_predictions, file, indent=2, ensure_ascii=False)

    # accu, wrong_predictions = eval_ins("/home/jiaxijzhang/llm_relevant_study/dataset/predictions_multi_data/multi_ins_predictions_greedy.json")
    # print(f"Accu: {accu}")
    # # with open(Path("/home/jiaxijzhang/llm_relevant_study/dataset/predictions_multi_data") / "pred_ins_greedy_with_oracle_wrong.json", "w", encoding="utf-8") as file:
    # #     json.dump(wrong_predictions, file, indent=2, ensure_ascii=False)