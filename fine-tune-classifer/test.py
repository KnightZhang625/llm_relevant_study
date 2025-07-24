# coding:utf-8
# Author: Jiaxin
# Date: 25-June-2025

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch

from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer_path = "/nfs1/jiaxinzhang/saved_for_checkpoint/qwen-classifier"
test_model_path = "/nfs1/jiaxinzhang/saved_for_checkpoint/qwen-classifier/checkpoint-479"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
test_model = AutoModelForSequenceClassification.from_pretrained(test_model_path, num_labels=2, id2label={0: "easy", 1: "hard"})

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
test_model.to(device)
classifier = pipeline("sentiment-analysis", model=test_model, tokenizer=tokenizer, device=0)

sample = "What's your favorite food?"

pred = classifier(sample)
# label: the class ID or name the model thinks is most likely (here 0).
# score: the model’s confidence for that prediction, i.e. the softmax probability assigned to the chosen label.
# [{'label': 0, 'score': 1.0}]
print(pred)