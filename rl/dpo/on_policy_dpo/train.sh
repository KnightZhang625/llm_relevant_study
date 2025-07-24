#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file /home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_distributed/configs/deepspeed_config.yaml ./example.py --yaml_config /home/jiaxijzhang/llm_relevant_study/rl/dpo/on_policy_dpo/configs/dpo-llama-3-1-8b-qlora.yaml