# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from transformers import AutoModelForSequenceClassification


# default hyperparameter choices
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

class DoRALayer(nn.Module):
    def __init__(self, linear, rank=4, alpha=8):
        super().__init__()

        self.weight = nn.Parameter(linear.weight, requires_grad=False)
        self.bias = nn.Parameter(linear.bias, requires_grad=False)

        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))
        self.alpha = alpha
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(linear.out_features, rank) * std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.in_features))
    
    def forward(self, x):
        lora = self.alpha * torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

for param in model.parameters():
    param.requires_grad = False

assign_dora = partial(DoRALayer, rank=lora_r, alpha=lora_alpha)

for layer in model.distilbert.transformer.layer:
    if lora_query:
        layer.attention.q_lin = assign_dora(layer.attention.q_lin)
    if lora_key:
        layer.attention.k_lin = assign_dora(layer.attention.k_lin)
    if lora_value:
        layer.attention.v_lin = assign_dora(layer.attention.v_lin)
    if lora_projection:
        layer.attention.out_lin = assign_dora(layer.attention.out_lin)
    if lora_mlp:
        layer.ffn.lin1 = assign_dora(layer.ffn.lin1)
        layer.ffn.lin2 = assign_dora(layer.ffn.lin2)
if lora_head:
    model.pre_classifier = assign_dora(model.pre_classifier)
    model.classifier = assign_dora(model.classifier)

# Check if linear layers are frozen
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total number of trainable parameters:", count_parameters(model))






