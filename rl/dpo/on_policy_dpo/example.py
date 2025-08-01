# coding:utf-8
# Author: Jiaxin
# Date: 09-July-2025

import sys
import logging
import os
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import TrlParser, ModelConfig, get_peft_config
from datasets import load_dataset
from trl import (
    DPOTrainer,
    DPOConfig,
    TrlParser,
    get_peft_config,
    ModelConfig,
)

from datasets import load_dataset

from config import (
    parse_training_args,
    ScriptArguments,
)

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
        dataset_id_or_path: str
        dataset_splits: str = "train"
        tokenizer_name_or_path: str = None

########################
# Setup logging
########################
log = logging.getLogger("on-policy-dpo")
log.setLevel(logging.INFO)
log.propagate = False
formatter = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%m-%d-%Y %H:%M:%S")
to_terminal = logging.StreamHandler(sys.stdout)
to_terminal.setFormatter(formatter)
log.addHandler(to_terminal)

########################
# Helper functions
########################

def get_checkpoint(training_args: DPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def dpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: DPOConfig
):
    #########################
    # Log parameters
    #########################
    log.info(f"Model parameters {model_args}")
    log.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    if script_args.dataset_id_or_path.endswith(".json"):
        train_dataset = load_dataset(
            "json", data_files=script_args.dataset_id_or_path, split="train"
        )
    else:
        train_dataset = load_dataset(
            script_args.dataset_id_or_path, split=script_args.dataset_splits
        )

    log.info(
        f"Loaded dataset with {len(train_dataset)} samples and the following features: {train_dataset.features}"
    )

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #####################
    # Prepare and format dataset
    #####################
    def format_dpo_sample(sample):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sample["system_prompt"]},
                {"role": "user", "content": sample["prompt"]},
            ],
            tokenize=False,
        )
        chosen = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["chosen"]}], tokenize=False,
        )
        rejected = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["rejected"]}], tokenize=False,
        )

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    
    # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
    train_dataset = train_dataset.map(
        format_dpo_sample, remove_columns=train_dataset.column_names
    )

    # remove all columns except chosen, rejected
    print(f"Columns: {train_dataset.features.keys()}")
    train_dataset = train_dataset.select_columns(["prompt", "chosen", "rejected"])

    #######################################
    # Load the model and/or reference model
    #######################################
    model_kwargs = dict(
        revision=model_args.model_revision,  # What revision from Huggingface to use, defaults to main
        trust_remote_code=model_args.trust_remote_code,  # Whether to trust the remote code, this also you to fine-tune custom architectures
        attn_implementation=model_args.attn_implementation,  # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=(
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        ),  # What torch dtype to use, defaults to auto
        use_cache=False if training_args.gradient_checkpointing else True,  # Whether
        low_cpu_mem_usage=(
            True
            if not (os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "false")
            else None
        ),  # Reduces memory usage on CPU for loading the model
    )

    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )

    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None

    # Policy Model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    # Checks wether we use adapters for reference model or not
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(
              model_args.model_name_or_path, **model_kwargs
        )
    else:
        model_ref = None

    #########################
    # Instantiate DPO trainer
    #########################
    trainer = DPOTrainer(
        model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        log.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    log.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    log.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    log.info("*** Save model ***")
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    log.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    log.info(f"Tokenizer saved to {training_args.output_dir}")

    log.info("*** Training complete! ***")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--yaml_config", type=str, required=True)
    args = parser.parse_args()

    yaml_config = args.yaml_config
    script_args, model_args, dpo_args = parse_training_args(config_file=yaml_config)

    dpo_function(model_args, script_args, dpo_args)