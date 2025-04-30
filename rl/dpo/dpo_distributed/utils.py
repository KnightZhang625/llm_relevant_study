# coding:utf-8

from datasets import Dataset
from transformers import AutoTokenizer

def convert_data_to_qwen_format(query: str, tokenizer: AutoTokenizer, return_tensors: bool=False):

    message = [
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": query},
    ]

    tokens = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
    )
    if not return_tensors:
        return tokens
    else:
        return tokenizer(tokens, add_special_tokens=False, return_tensors="pt")


def process_dataset(raw_dataset: Dataset, tokenizer: AutoTokenizer, shuffle: bool=True, test_ratio: float=0.1):
    formatted_datasets = [
        {
            "prompt": convert_data_to_qwen_format(data["question"], tokenizer),
            "chosen": data["chosen"],
            "rejected": data["rejected"],
        }
        for data in raw_dataset
    ]

    processed_dataset = Dataset.from_list(formatted_datasets)
    if shuffle:
        processed_dataset = processed_dataset.shuffle()
    processed_dataset = processed_dataset.train_test_split(test_size=test_ratio)

    return processed_dataset["train"], processed_dataset["test"]