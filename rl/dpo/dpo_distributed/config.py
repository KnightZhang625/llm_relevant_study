# coding:utf-8

from dataclasses import dataclass
from transformers import HfArgumentParser
from trl import DPOConfig, ModelConfig

@dataclass
class DatasetArgs:

    raw_data_path: str

def _parse_from_yaml(parser: HfArgumentParser, config_file: str):
    return parser.parse_yaml_file(config_file)

def parse_training_args(config_file: str):
    parser = HfArgumentParser((DatasetArgs, ModelConfig, DPOConfig))
    return _parse_from_yaml(parser, config_file)

if __name__ == "__main__":
    data_args, model_args, training_args = parse_training_args(config_file="/home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_distributed/configs/dpo_qwen_2.5_0.5b_instruct_lora_train.yaml")

    print(data_args)
    print(model_args)
    print(training_args)