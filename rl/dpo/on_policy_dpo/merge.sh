#!/usr/bin/bash

ARGS="""--model_name_or_path /nfs1/jiaxinzhang/models/llama-3-1-8b-math-orca-spectrum-10k-ep1 \
        --peft_model_id /nfs1/jiaxinzhang/saved_for_checkpoint/dpo-llama-3-1-8b-math-ep3 \
        --output_dir /nfs1/jiaxinzhang/saved_for_checkpoint/merged-dpo-llama-3-1-8b-math-ep3 \
"""

python merge.py ${ARGS}