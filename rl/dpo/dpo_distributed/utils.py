# coding:utf-8

from datasets import Dataset
from transformers import AutoTokenizer

def convert_data_to_qwen_format(query: str, chosen: str, rejected: str, tokenizer: AutoTokenizer=None, return_tensors: bool=False):
    """Convert query (plain str) to the chat_template format."""

    query_message = [
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": query}
    ]
    chosen_message = []
    if chosen is not None:
        chosen_message = [{"role": "assistant", "content": chosen}]
    rejected_message = []
    if rejected is not None:
        rejected_message = [{"role": "assistant", "content": rejected}]

    if not return_tensors:
        if tokenizer is None:
            # NOTE: Format 1: we return the conversation format directly, the DPOTrainer will handle the chat template.
            return {
                "prompt": query_message,
                "chosen": chosen_message,
                "rejected": rejected_message,
            }
        else:
            # NOTE: Format 2: we apply chat template by ourself
            prompt_chat_template = tokenizer.apply_chat_template(
                query_message,
                tokenize=False,
                add_generation_prompt=True,
            )

            chosen_chat_template = tokenizer.apply_chat_template(
                query_message + chosen_message,
                tokenize=False,
            )[len(prompt_chat_template):]

            rejected_chat_template = tokenizer.apply_chat_template(
                query_message + rejected_message,
                tokenize=False,
            )[len(prompt_chat_template):]

            return {
                "prompt": prompt_chat_template,
                "chosen": chosen_chat_template,
                "rejected": rejected_chat_template,
            }
    else:
        # we can use "tokenize=True, return_tensors='pt' ", however, this will only return the input_ids of the message, lacking "attention_mask",
        # will cause the alert: "The attention mask is not set and cannot be inferred from input because pad token is same as eos token. 
        #   As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
        # Therefore, we use tokenizer to get attention_mask.
        tokens = tokenizer.apply_chat_template(
            query_message,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(tokens, add_special_tokens=False, return_tensors="pt")

def process_dataset(raw_dataset: Dataset, tokenizer: AutoTokenizer, shuffle: bool=True, test_ratio: float=0.1):
    formatted_datasets = [
        convert_data_to_qwen_format(query=data["question"], chosen=data["chosen"], rejected=data["rejected"], tokenizer=None, return_tensors=False)
        for data in raw_dataset
    ]

    processed_dataset = Dataset.from_list(formatted_datasets)
    if shuffle:
        processed_dataset = processed_dataset.shuffle()
    processed_dataset = processed_dataset.train_test_split(test_size=test_ratio)

    return processed_dataset["train"], processed_dataset["test"]