#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /home/jiaxijzhang/llm_relevant_study/rl/dpo/dpo_distributed/saved_models \
    --served-model-name qwen_dpo \
    --max-model-len=512 \
    --port 56001 \
    --dtype=bfloat16