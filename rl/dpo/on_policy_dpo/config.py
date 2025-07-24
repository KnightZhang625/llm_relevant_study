# coding:utf-8

from dataclasses import dataclass
from trl import (
    DPOConfig, 
    ModelConfig,
)
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None

def _parse_from_yaml(parser: HfArgumentParser, config_file: str):
    return parser.parse_yaml_file(config_file)

def parse_training_args(config_file: str):
    parser = HfArgumentParser((ScriptArguments, ModelConfig, DPOConfig))
    return _parse_from_yaml(parser, config_file)