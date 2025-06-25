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

raw_dataset = load_dataset("/home/jiaxijzhang/llm_relevant_study/dataset/llm_router_dataset-synth")
print(f"Train dataset size: {len(raw_dataset['train'])}")
print(f"Test dataset size: {len(raw_dataset['test'])}")

model_path = "/nfs1/jiaxinzhang/models/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.model_max_length = 512

def tokenize(batch):
    return tokenizer(batch["prompt"], padding="max_length", truncation=True, return_tensors="pt")

# Tokenize dataset
raw_dataset =  raw_dataset.rename_column("label", "labels") # to match Trainer
tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["prompt"])

labels = list(set(tokenized_dataset["train"]["labels"]))
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
        labels, predictions, labels=labels, pos_label=1, average="weighted",
    )
    return {"f1": float(score) if score == 1 else score}

training_args = TrainingArguments(
    output_dir="/nfs1/jiaxinzhang/saved_for_checkpoint/bert-classifier",
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
    run_name="bert-classifier-1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# # Start training
# trainer.train()

# /nfs1/jiaxinzhang/saved_for_checkpoint/bert-classifier/checkpoint-240
# load model from huggingface.co/models using our repository id
tokenizer_path = "/nfs1/jiaxinzhang/saved_for_checkpoint/bert-classifier"
test_model_path = "/nfs1/jiaxinzhang/saved_for_checkpoint/bert-classifier/checkpoint-480"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
test_model = AutoModelForSequenceClassification.from_pretrained(
   test_model_path, num_labels=num_labels, label2id=label2id, id2label=id2label,
)
test_model.to("cuda")
classifier = pipeline("sentiment-analysis", model=test_model, tokenizer=tokenizer, device=0)

sample = "How does the structure and function of plasmodesmata affect cell-to-cell communication and signaling in plant tissues, particularly in response to environmental stresses?"

pred = classifier(sample)
print(pred)