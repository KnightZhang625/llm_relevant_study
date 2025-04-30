# coding:utf-8

import os
os.environ["WANDB_PROJECT"] = "dpo-qwen"
import sys
import logging

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, ModelConfig, DPOTrainer
from config import parse_training_args, DatasetArgs
from utils import process_dataset

log = logging.getLogger("dpo-qwen-training-log")
log.setLevel(logging.INFO)
log.propagate = False
formatter = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%m-%d-%Y %H:%M:%S")

def train(data_args: DatasetArgs, model_args: ModelConfig, training_args: DPOConfig):
    # ----- load model, tokenizer ----- #
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True, padding_side="left")

    # ----- load dataset ----- #
    raw_dataset = load_dataset(data_args.raw_data_path)["train"]
    train_dataset, test_dataset = process_dataset(raw_dataset, tokenizer)

    # ----- Lora config ----- #
    peft_model = None
    if model_args.use_peft:
        log.info("Use LoRA.")
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=model_args.lora_task_type,
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.config.use_cache = False
    
    # ----- DPO trainer ----- #
    trainer = DPOTrainer(
        model=peft_model if model_args.use_peft else model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer
    )
    log.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    trainer.train()
    log.info("*** Training complete ***")

    log.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    log.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    log.info(f"Tokenizer saved to {training_args.output_dir}")
    log.info("*** Saving complete! ***")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--yaml_config", type=str, required=True)
    args = parser.parse_args()

    yaml_config = args.yaml_config
    data_args, model_args, training_args = parse_training_args(config_file=yaml_config)

    Path(training_args.output_dir).mkdir(exist_ok=True)
    to_disk = logging.FileHandler(os.path.join(training_args.output_dir, training_args.run_name + ".log"))
    to_disk.setLevel(logging.DEBUG)
    to_disk.setFormatter(formatter)
    log.addHandler(to_disk)

    to_terminal = logging.StreamHandler(sys.stdout)
    to_terminal.setLevel(logging.INFO)
    to_terminal.setFormatter(formatter)
    log.addHandler(to_terminal)

    train(data_args, model_args, training_args)