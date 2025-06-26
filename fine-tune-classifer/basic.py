# coding:utf-8
# Author: Jiaxin
# Date: 25-June-2025

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_PROJECT"] = "bert-classifier"  # optional: name your project
import numpy as np
from sklearn.metrics import f1_score

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

raw_dataset = load_dataset("dair-ai/emotion", "split")

model_path = "/nfs1/jiaxinzhang/models/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.model_max_length = 512

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="pt")

# Tokenize dataset
raw_dataset =  raw_dataset.rename_column("label", "labels") # to match Trainer
tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])

# label2id, id2label是无所谓的, 和数据中labels无关系,
# label2id: 不知道干嘛的
# id2label: 讲预测的id转为label
# labels = list(set(tokenized_dataset["train"]["labels"]))
labels = ["sadness", "joy", "anger", "fear", "love", "surprise"]
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=num_labels, label2id=label2id, id2label=id2label,
)
model.to("cuda")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
        labels, predictions, average="weighted",
    )
    return {"f1": float(score) if score == 1 else score}

training_args = TrainingArguments(
    output_dir="/nfs1/jiaxinzhang/saved_for_checkpoint/bert-classifier-multi",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    num_train_epochs=5,
    bf16=True,
    optim="adamw_torch_fused",
    logging_strategy="steps",
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",     # must be the name of metric returned by compute_metrics
    report_to="wandb",
    run_name="bert-classifier-multi-classes",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# # Start training
# trainer.train()

# /nfs1/jiaxinzhang/saved_for_checkpoint/bert-classifier/checkpoint-240
# load model from huggingface.co/models using our repository id
tokenizer_path = "/nfs1/jiaxinzhang/models/ModernBERT-large"
test_model_path = "/nfs1/jiaxinzhang/saved_for_checkpoint/bert-classifier-multi/checkpoint-250"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# 模型已经保存了label2id=label2id, id2label=id2label,
test_model = AutoModelForSequenceClassification.from_pretrained(
   test_model_path, num_labels=num_labels
)
test_model.to("cuda")
classifier = pipeline("sentiment-analysis", model=test_model, tokenizer=tokenizer, device=0)

sample = raw_dataset["test"][0]
pred = classifier(sample["text"])
print(f"Pred: {pred}")
print(f"Gold: {id2label[sample["labels"]]}")