# coding:utf-8

from dataclasses import dataclass
from transformers import HfArgumentParser
from trl import GRPOConfig, ModelConfig
# from peft import LoraConfig

########################
# Custom dataclasses
########################
@dataclass
class PathArguments:

    data_path: str = "/home/jiaxijzhang/llm_relevant_study/rl/grpo/datasets/Qwen2.5-3B-Instruct-Countdown-Tasks-3to4"
    data_split: str = "train"

def _parse_from_yaml(parser: HfArgumentParser, config_path: str):
    return parser.parse_yaml_file(config_path)

def parse_model_path_args(config_path: str):
    parser = HfArgumentParser((ModelConfig, PathArguments, GRPOConfig))
    return _parse_from_yaml(parser, config_path)

if __name__ == "__main__":
    model_args, path_args, grpo_args = parse_model_path_args("/home/jiaxijzhang/llm_relevant_study/rl/grpo/distributed_code/data_model_config.yaml")
