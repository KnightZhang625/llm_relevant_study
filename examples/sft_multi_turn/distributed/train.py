# coding:utf-8
# Author: Jiaxin
# Date: 17-June-2025

import os
os.environ["WANDB_PROJECT"] = "sft-lora-qwen-multi-turn-dialogue"
import sys
import logging
import json

from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import Dataset

from config import (
    DatasetArgs,
    ModelConfigSelf,
    LoraConfigSelf,
    parse_training_args,
)

log = logging.getLogger("sft-lora-qwen-multi-turn-dialogue-log")
log.setLevel(logging.INFO)
log.propagate = False
formatter = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%m-%d-%Y %H:%M:%S")

def load_dataset(data_path: str) -> list[dict]:
    log.info(f"Read data from: {data_path}")
    datas = []
    bad_lines = []
    with open(data_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, 1):
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    conversations = record.get("conversations")
                    if isinstance(conversations, list) and all(isinstance(m, dict) for m in conversations):
                        datas.append(record)
                    else:
                        bad_lines.append((i, conversations))
                except json.JSONDecodeError as e:
                    bad_lines.append((i, f"JSON error: {e}"))

    log.info(f"✅ Loaded {len(datas)} valid records.")
    log.info(f"⚠️ Skipped {len(bad_lines)} invalid records.")
    
    for i, issue in bad_lines:
        log.debug(f"  Line {i}: {issue}")

    return datas

def convert_into_conversation_format(data):
    raw_data = data["conversations"]
    messages = [{"role": "system", "content": "你是一个淘宝AI助手。"}]
    for conv in raw_data:
        messages.append(
            {
                "role": str(conv["from"]),
                "content": str(conv["value"]),
            }
        )
    return {"messages": messages}

def train(data_args: DatasetArgs, model_args: ModelConfigSelf, lora_args: LoraConfigSelf, sft_args: SFTConfig):
    # ----- load tokenizer, model ----- #
    log.info(f"Load tokenizer and model from: {model_args.model_name_or_path}")
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
    raw_data = load_dataset(data_args.raw_data_path)
    log.info(f"In total: {len(raw_data)} raw datas.")
    formatted_data = [convert_into_conversation_format(data) for data in raw_data]
    log.info(f"Example from formatted data: \n{formatted_data[0]}\n")
    dataset = Dataset.from_list(formatted_data)

    # ----- lora config ----- #
    peft_config = None
    if lora_args.use_lora:
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

    to_sdtout = logging.StreamHandler(sys.stdout)
    to_sdtout.setLevel(logging.INFO)
    to_sdtout.setFormatter(formatter)
    log.addHandler(to_sdtout)

    train(dataset_args, model_config, lora_config, sft_config)