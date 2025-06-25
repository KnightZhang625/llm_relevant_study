# coding:utf-8
# Author: Jiaxin
# Date: 25-June-2025

import torch
import numpy as np

from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from functools import partial
from sklearn.metrics import f1_score
from config import (
    DatasetArgs,
    ModelConfigSelf,
    parse_training_args,
)

def tokenize(batch: dict, tokenizer: AutoTokenizer):
    return tokenizer(batch["prompt"], padding="max_length", truncation=True, return_tensors="pt")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(labels, predictions, pos_label=1, average="weighted")
    return {"f1": float(score) if score == 1 else score}

def main(dataset_args: DatasetArgs, model_config: ModelConfigSelf, training_args: TrainingArguments):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # load dataset
    raw_dataset = load_dataset(dataset_args.raw_data_path)
    print(f"Train dataset size: {len(raw_dataset['train'])}")
    print(f"Test dataset size: {len(raw_dataset['test'])}")
    raw_dataset =  raw_dataset.rename_column("label", "labels") # to match Trainer

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.model_max_length = model_config.model_max_length
    tokenize_fn = partial(tokenize, tokenizer=tokenizer)

    # tokenized dataset
    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["prompt"])
    labels = list(set(tokenized_dataset["train"]["labels"]))
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    
    # create model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=num_labels, label2id=label2id, id2label=id2label,
    )
    model.to(device)

    # set trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--yaml_config", type=str, required=True)
    args = parser.parse_args()

    dataset_args, model_config, training_args = parse_training_args(args.yaml_config)

    main(dataset_args, model_config, training_args)