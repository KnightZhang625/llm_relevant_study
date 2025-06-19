#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file ./ds.yaml ./train.py --yaml_config /home/jiaxijzhang/llm_relevant_study/examples/sft_multi_turn/distributed/config.yaml