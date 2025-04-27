# coding:utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["WANDB_PROJECT"] = "gemma-dolly-chat"  # optional: name your project
os.environ["WANDB_NAME"] = "gemma-7b-dolly-sft-run-1"  # set the specific run name

import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


dataset = load_dataset("parquet", data_files="/home/jiaxijzhang/llm_relevant_study/pipeline/dolly-15k-oai-style/data/train-00000-of-00001-54e3756291ca09c6.parquet", split="train")
print(dataset[3]["messages"])

# Hugging Face model id
model_id = "/nfs3/nlp_common/LLM_Models/DeepSeek-R1-Distill-Qwen-7B"
tokenizer_id = "/nfs3/nlp_common/LLM_Models/DeepSeek-R1-Distill-Qwen-7B"

# # BitsAndBytesConfig int-4 config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )
 
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="auto",
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.padding_side = 'right' # to prevent warnings
# print(tokenizer.apply_chat_template(dataset[3]["messages"], tokenize=False))

# peft_config = LoraConfig(
#     lora_alpha=8,
#     lora_dropout=0.05,
#     r=6,
#     bias="none",
#     target_modules="all-linear",
#     task_type="CAUSAL_LM"
# )

# args = TrainingArguments(
#     output_dir="gemma-7b-dolly-chatml",     # directory to save and repository id
#     num_train_epochs=3,                     # number of training epochs
#     per_device_train_batch_size=2,          # batch size per device during training
#     gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
#     gradient_checkpointing=False,            # use gradient checkpointing to save memory
#     optim="adamw_torch_fused",              # use fused adamw optimizer
#     logging_steps=10,                       # log every 10 steps
#     save_strategy="epoch",                  # save checkpoint every epoch
#     bf16=True,                              # use bfloat16 precision
#     tf32=True,                              # use tf32 precision
#     learning_rate=2e-4,                     # learning rate, based on QLoRA paper
#     max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
#     warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
#     lr_scheduler_type="constant",           # use constant learning rate scheduler
#     push_to_hub=False,                      # push model to hub
#     report_to="wandb",                # report metrics to tensorboard
# )

# max_seq_length = 1512 # max sequence length for model and packing of the dataset
 
# trainer = SFTTrainer(
#     model=model,
#     args=args,
#     train_dataset=dataset,
#     peft_config=peft_config,
#     max_seq_length=max_seq_length,
#     tokenizer=tokenizer,
#     packing=True,
#     dataset_kwargs={
#         "add_special_tokens": False, # We template with special tokens
#         "append_concat_token": False, # No need to add additional separator token
#     }
# )

# # start training, the model will be automatically saved to the hub and the output directory
# trainer.train()
 
# # save model
# trainer.save_model()

adapter = PeftModel.from_pretrained(model, "/home/jiaxijzhang/llm_relevant_study/pipeline/gemma-7b-dolly-chatml")
merged_model = adapter.merge_and_unload()
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
test_data = [dataset[3]["messages"][0]]
print(f"Que: {test_data}")
prompt = pipe.tokenizer.apply_chat_template(test_data, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
print()
print(f"Out: {outputs}")

print()
print()
print()
prompt = tokenizer.apply_chat_template(test_data, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(
    merged_model.generate(prompt.to(merged_model.device), max_new_tokens=1024)[0], 
    skip_special_tokens=True))