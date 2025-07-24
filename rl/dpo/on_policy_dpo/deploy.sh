#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
#     --model /nfs1/jiaxinzhang/saved_for_checkpoint/qwen2.5-7b-ins-chat-multi-round/checkpoint-327 \
#     --served-model-name qwen-7b \
#     --max-model-len=2048 \
#     --port 56001 \
#     --dtype=bfloat16


CUDA_VISIBLE_DEVICES=0 vllm serve /nfs1/jiaxinzhang/saved_for_checkpoint/merged-dpo-llama-3-1-8b-math-ep3 \
  --dtype auto \
  --served-model-name llama-8b \
  --port 56001