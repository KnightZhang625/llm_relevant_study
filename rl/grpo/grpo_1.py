# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "grpo-qwen"  # optional: name your project
os.environ["WANDB_NAME"] = "grpo-qwen-2.5-1.5b-instruct-run-1"  # set the specific run name
import re

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig

dataset_path = "/home/jiaxijzhang/llm_relevant_study/rl/grpo/datasets/Qwen2.5-3B-Instruct-Countdown-Tasks-3to4"
dataset = load_dataset(dataset_path, split="train")
dataset = dataset.shuffle(seed=42).select(range(50000))

tokenizer = AutoTokenizer.from_pretrained("/nfs1/jiaxinzhang/models/Qwen2.5-1.5B-Instruct")

def generate_r1_prompt(numbers, target):
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        {
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]), num_proc=8)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []
    
    for completion, gt in zip(completions, target):

        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # Check if the format is correct
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

            match = re.search(regex, completion, re.DOTALL)
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue
        
            # Evaluate the equation with restricted globals and locals
            # It safely evaluates the string in equation.
            # The evaluated code can't use any built-in Python functions or modules.
            # Only pure expressions like "2 + 2", or "3 * (4 + 5)" will work.
            result = eval(equation, {"__builtins__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    
    return rewards

# our model we are going to use as policy 
model_config = ModelConfig(
    model_name_or_path="/nfs1/jiaxinzhang/models/Qwen2.5-1.5B-Instruct",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    use_peft=True,
    load_in_4bit=False,
)

# Hyperparameters
training_args = GRPOConfig(
    output_dir="saved_models/qwen-r1-aha-moment",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=8,
    logging_steps=10,
    max_steps=100,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=256,
    max_completion_length=1024, # max length of the generated output for our solution
    num_generations=2,
    beta=0.001,
    report_to="wandb"
)

trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, equation_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)

# Train and push the model to the Hub
trainer.train()
# Save model
trainer.save_model(training_args.output_dir)