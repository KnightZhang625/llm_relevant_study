#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file ./deepspeed_config.yaml ./train.py --data_model_config ./data_model_config.yaml