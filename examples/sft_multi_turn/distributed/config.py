# coding:utf-8
# Author: Jiaxin
# Date: 17-June-2025

from dataclasses import dataclass
from trl import SFTConfig
from transformers import HfArgumentParser

@dataclass
class DatasetArgs:

    raw_data_path: str

@dataclass
class ModelConfigSelf:
    model_name_or_path: str
    torch_dtype: str
    attn_implementation: str

@dataclass
class LoraConfigSelf:
    use_lora: bool
    r: int
    lora_alpha: int
    lora_dropout: float
    
    target_modules: str
    task_type: str

    bias: str="none"

def _parse_from_yaml_file(parser: HfArgumentParser, config_file: str):
    return parser.parse_yaml_file(config_file)

def parse_training_args(config_file: str):
    parser = HfArgumentParser((DatasetArgs, ModelConfigSelf, LoraConfigSelf, SFTConfig))
    return _parse_from_yaml_file(parser, config_file)