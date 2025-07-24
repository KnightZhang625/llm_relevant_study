# coding:utf-8

import re
import json
from pathlib import Path

convert = {
    "标记物资": "给AI标记指定物资",
    "封烟": "封烟区域"
}

def check_ans(pred, fuzzy_match=False):
    gold_intent, pred_intent = pred["intent"], pred["pred_intent"]
    pred_intent = convert.get(pred_intent, pred_intent)
    gold_item, pred_item = pred["item"], pred["pred_item"]

    if gold_intent != pred_intent:
        return False

    if (gold_item is None or gold_item.strip() == "") and pred_item == "空的":
        return True
    elif gold_item == pred_item:
        return True
    else:
        if fuzzy_match:
            if gold_item is not None and pred_item is not None:
                if gold_item in pred_item or pred_item in gold_item:
                    return True
        return False

def test_one_file(path):
    predictions = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                predictions.append(json.loads(line.strip()))
    
    scores = []
    wrong_predictions = []
    for pred in predictions:
        if check_ans(pred):
            scores.append(1)
        else:
            scores.append(0)
            wrong_predictions.append(pred)
   
    return sum(scores) / len(scores), wrong_predictions

def main(pred_dir):
    pred_files = Path(pred_dir).glob("*.jsonl")
    for file in pred_files:
        intent = re.sub("pred_", "", file.stem)
        try:
            score, wrong_predictions = test_one_file(file)
        except Exception as e:
            print(file)
            print(e)
            exit()
        print(f"Intent: {intent}, Score: {score:.2f}")
        with open(Path(pred_dir) / f"pred_{intent}_wrong.jsonl", "w", encoding="utf-8") as w_file:
            for line in wrong_predictions:
                w_file.write(json.dumps(line, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main("/home/jiaxijzhang/llm_relevant_study/dataset/predictions")

    # a = {'data': '走上车我要走了', 'intent': '上车', 'item': '', 'template': '', 'pred_intent': '上车', 'pred_item': '空的', 'output': '你说的是 上车 指令，其中关键信息是 空的。'}
    # print(check_ans(a))