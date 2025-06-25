# coding:utf-8
# Author: Jiaxin
# Date: 25-June-2025

from dataclasses import dataclass
from transformers import TrainingArguments, HfArgumentParser

@dataclass
class DatasetArgs:

    raw_data_path: str

@dataclass
class ModelConfigSelf:
    model_name_or_path: str
    model_max_length: int

def _parse_from_yaml_file(parser: HfArgumentParser, config_file: str):
    return parser.parse_yaml_file(config_file)

def parse_training_args(config_file: str):
    parser = HfArgumentParser([DatasetArgs, ModelConfigSelf, TrainingArguments])
    return _parse_from_yaml_file(parser, config_file)