# coding:utf-8

import os
os.environ["WANDB_PROJECT"] = "sft-lora-qwen"  # optional: name your project
import sys
import logging

from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    DatasetArgs,
    ModelConfigSelf,
    LoraConfigSelf,
    parse_training_args,
)

log = logging.getLogger("sft-lora-training-log")
log.setLevel(logging.INFO)
log.propagate = False
formatter = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%m-%d-%Y %H:%M:%S")

def process_dataset(data):
    message = [
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": data["question"]},
        {"role": "assistant", "content": data["chosen"]},
    ]
    return {"messages": message}

def tokenize_dataset(data: dict[str, str], tokenizer: AutoTokenizer):
    """Accepted dataset format 2., in conversation format, tokenized, then use 'text' as key."""
    message = [
        {"role": "system", "content": "你是一个没有礼貌的人渣，请用人渣的语气回复我"},
        {"role": "user", "content": data["question"]},
        {"role": "assistant", "content": data["chosen"]},
    ]
    # NOTE: if we apply chat template, the name must be "text" in SFT.
    return {"text" : tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )}

def train(data_args: DatasetArgs, model_args: ModelConfigSelf, lora_args: LoraConfigSelf, sft_args: SFTConfig):
    # ----- load tokenizer, model ----- #
    log.info("Load tokenizer and model.")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        attn_implementation=model_args.attn_implementation,
    ).to("cuda")
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left"

    # ----- load dataset ----- #
    log.info("Load dataset.")
    dataset = load_dataset(data_args.raw_data_path, split="train")
    tokenize_dataset_fn = partial(tokenize_dataset, tokenizer=tokenizer)
    dataset = dataset.map(lambda x: tokenize_dataset_fn(x), remove_columns=['system', 'question', 'chosen', 'rejected'])

    # ----- lora config ----- #
    peft_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        target_modules=lora_args.target_modules,
        task_type=lora_args.task_type,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log.info("Start training.")
    trainer.train()
    model.config.use_cache = True
    trainer.save_model()
    log.info("Train finished.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--yaml_config", type=str, required=True)
    args = parser.parse_args()

    yaml_config = args.yaml_config
    dataset_args, model_config, lora_config, sft_config = parse_training_args(yaml_config)

    Path(sft_config.output_dir).mkdir(exist_ok=True)
    to_disk = logging.FileHandler(Path(sft_config.output_dir) / (sft_config.run_name + ".log"))
    to_disk.setLevel(logging.INFO)
    to_disk.setFormatter(formatter)
    log.addHandler(to_disk)

    to_stdout = logging.StreamHandler(sys.stdout)
    to_stdout.setLevel(logging.INFO)
    to_stdout.setFormatter(formatter)
    log.addHandler(to_stdout)

    train(dataset_args, model_config, lora_config, sft_config)