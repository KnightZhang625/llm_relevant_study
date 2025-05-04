# coding:utf-8

import os
os.environ["WANDB_PROJECT"] = "grpo-qwen"
import logging

from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from trl import GRPOConfig, ModelConfig, get_peft_config, GRPOTrainer
from peft import LoraConfig, get_peft_model

from config import parse_model_path_args, PathArguments
from utils import generate_r1_prompt, format_reward_func, equation_reward_func

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

def train(model_args: ModelConfig, path_args: PathArguments, training_args: GRPOConfig):
    logging.info(f"Model Parameters:\n{model_args}")
    logging.info(f"Training Parameters:\n{training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ###############
    # Load datasets
    ###############
    dataset = load_dataset(path_args.data_path, split="train")
    dataset = dataset.shuffle(seed=42).select(range(100))

    generate_r1_prompt_with_tokenizer = partial(generate_r1_prompt, tokenizer=tokenizer)
    dataset = dataset.map(lambda x: generate_r1_prompt_with_tokenizer(x["nums"], x["target"]), num_proc=8)
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    ###############
    # Load model
    ###############
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        torch_dtype=model_args.torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    model.to("cuda")  # No need to call model.to("cuda") — Trainer will handle it. Do: `model = accelerator.prepare(model)`
    model.config.use_cache = False

    ###############
    # Lora config
    ###############
    peft_model = None
    if model_args.use_peft:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            task_type=model_args.lora_task_type,
            bias="none",
        )
        peft_model = get_peft_model(model, peft_config)

    #########################
    # Instantiate DPO trainer
    #########################
    trainer = GRPOTrainer(
        model=model if not model_args.use_peft else peft_model,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    ###############
    # Training loop
    ###############
    last_checkpoint = None
    # if Path(training_args.output_dir).is_dir():
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    # TrainOutput:
    #   - global_step: how many steps were trained. 
    #   - training_loss: total loss.
    #   - metrics: a dictionary with training metrics like total steps, loss, etc.
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    ###############
    # Metrics
    ###############
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)   # to wandb
    trainer.save_metrics('train', metrics)  # Saves the metrics as a JSON file under the output_dir.
    trainer.save_state()

    logger.info("*** Training complete ***")

    #################
    # Save model
    #################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    logger.info("*** Training complete! ***")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_model_config", type=str, required=True)
    args = parser.parse_args()

    data_model_config_path = args.data_model_config
    model_args, path_args, grpo_args = parse_model_path_args(data_model_config_path)

    
    train(model_args, path_args, grpo_args)