#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 ./multi_gpus.py --yaml_config /home/jiaxijzhang/llm_relevant_study/LoRA/lora_trainer/distributed_version/config.yaml