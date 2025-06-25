#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes 3 distributed_train.py --yaml_config config.yaml