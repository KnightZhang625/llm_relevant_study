# coding:utf-8
# Author: Jiaxin
# Date: 16-June-2025

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "sft-lora-qwen-multi-turn"  # optional: name your project
# import wandb
# wandb.login()

import json
import torch
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer, setup_chat_format

def convert_into_conversation_format(data):
    raw_data = data["conversations"]
    messages = [{"role": "system", "content": "你是一个AI助手。"}]
    for conv in raw_data:
        messages.append(
            {
                "role": str(conv["from"]),
                "content": str(conv["value"]),
            }
        )
    return {"messages": messages}

def load_dataset():
    datas = []
    bad_lines = []
    data_path = "/home/jiaxijzhang/llm_relevant_study/dataset/processed_ins_chat.json"
    with open(data_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, 1):
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    conversations = record.get("conversations")
                    if isinstance(conversations, list) and all(isinstance(m, dict) for m in conversations):
                        datas.append(record)
                    else:
                        bad_lines.append((i, conversations))
                except json.JSONDecodeError as e:
                    bad_lines.append((i, f"JSON error: {e}"))

    print(f"✅ Loaded {len(datas)} valid records.")
    print(f"⚠️ Skipped {len(bad_lines)} invalid records.")
    
    for i, issue in bad_lines:
        print(f"  Line {i}: {issue}")

    return datas

if __name__ == "__main__":
    ds = load_dataset()
    print(f"In total: {len(ds):,}.")

    formatted_data = [convert_into_conversation_format(line) for line in ds]
    print(formatted_data[0])

    dataset = Dataset.from_list(formatted_data)
    print(dataset[0])

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model_path = "/nfs1/jiaxinzhang/models/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    ) 
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir="saved_models",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=1e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.3,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        max_length=1024,
        packing=True,
        report_to="wandb",
        run_name="sft-lora-qwen-2.5-0.5b-instruct-multi-turn-run-1",
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()

    # merged_model = PeftModel.from_pretrained(
    #     model,
    #     "/home/jiaxijzhang/llm_relevant_study/examples/sft_multi_turn/saved_models"
    # ).to(device)
    # merged_model = merged_model.merge_and_unload()

    # test_data = {"conversations": [{"from": "user", "value": "快点发货谢谢"}, {"from": "assistant", "value": "您好，我是小店【智能助理】。现在是客服休息时间，由我为您服务~"}, {"from": "user", "value": "我已经拼单成功3小时，能尽快帮我发货吗？"}]}
    # test_message = convert_into_conversation_format(test_data)
    # tokens = tokenizer.apply_chat_template(
    #     test_message["messages"],
    #     tokenize=False,
    #     add_generation_prompt=True,
    # )
    # inputs = tokenizer(tokens, add_special_tokens=False, return_tensors="pt")

    # output = merged_model.generate(
    #     input_ids=inputs["input_ids"].to(device),
    #     attention_mask=inputs["attention_mask"].to(device),
    #     max_length=512,
    #     temperature=0.7,
    #     top_p=0.9,
    #     do_sample=True,
    #     eos_token_id=tokenizer.eos_token_id,
    # )

    # print(tokenizer.decode(output[0][inputs["input_ids"].size(1):], skip_special_tokens=True))