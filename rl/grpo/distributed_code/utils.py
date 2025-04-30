# coding:utf-8

import re

def generate_r1_prompt(numbers, target, tokenizer):
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

    return {
        "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
        "target": target,
    }

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