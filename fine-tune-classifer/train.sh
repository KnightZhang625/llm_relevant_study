#!/usr/bin/bash

# CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes 1 distributed_train.py --yaml_config config.yaml

CUDA_VISIBLE_DEVICES=7 python distributed_train.py --yaml_config config.yaml