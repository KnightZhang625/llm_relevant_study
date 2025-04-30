#!/usr/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file /home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_distributed/configs/deepspeed_config.yaml ./train.py --yaml_config /home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_distributed/configs/dpo_qwen_2.5_0.5b_instruct_lora_train.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 ./train.py --yaml_config /home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_distributed/configs/dpo_qwen_2.5_0.5b_instruct_lora_train.yaml