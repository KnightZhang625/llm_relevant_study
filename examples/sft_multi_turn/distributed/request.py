import re
import json
import concurrent.futures as futures
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

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
        # messages=[
        #     {"role": "system", "content": "你是一个指令和闲聊助手，请根据用户的输入判断用户输入是闲聊还是指令。"},
        #     {"role": "user", "content": "三把装备不然那你什么枪给我吧，有直角前握把吗。"},
        # ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=2048,
    )
    # print("Chat response:", chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content

def parse_instruct_ans(ans) -> list:
    if "你说的是" not in ans and "关键信息是" not in ans:
        return ["", ""]

    matches = re.findall(r"你说的是 (.*?) 指令.*?关键信息是 (.*?)。", ans)

    if matches:
        command, keyword = matches[0]
        return [command, keyword]
    else:
        return ["", ""]

def test_one_instruct_file(datas):
    for data in tqdm(datas, total=len(datas), desc=f"{datas[0]["intent"]}", leave=False):
        messages=[
            {"role": "system", "content": "你是一个指令和闲聊助手，请根据用户的输入判断用户输入是闲聊还是指令。"},
            {"role": "user", "content": data["data"]},
        ]
        ans = request_api(messages)
        parsed_res = parse_instruct_ans(ans)
        data["pred_intent"] = parsed_res[0]
        data["pred_item"] = parsed_res[1]
        data["output"] = ans
    
    return datas

def load_ins(path):
    intent = re.sub("test_", "", path.stem)
    test_datas = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                test_datas.append(json.loads(line.strip()))
    return intent, test_datas

def predict(path):
    intent, test_datas = load_ins(path)
    predictions = test_one_instruct_file(test_datas)
    return {intent: predictions}

def main(test_dir, save_dir):
    all_paths = Path(test_dir).glob("*.jsonl")
    Path(save_dir).mkdir(exist_ok=True)
    threads = []
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        for path in all_paths:
            threads.append(executor.submit(predict, path))

        for future in futures.as_completed(threads):
            predictions = future.result()
            with open(Path(save_dir) / f"pred_{list(predictions.keys())[0]}.jsonl", "w", encoding="utf-8") as file:
                for pred in list(predictions.values())[0]:
                    file.write(json.dumps(pred, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main("/home/jiaxijzhang/llm_relevant_study/dataset/test_data", "/home/jiaxijzhang/llm_relevant_study/dataset/predictions")