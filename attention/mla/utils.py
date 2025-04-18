# coding:utf-8
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import time
from tqdm import tqdm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, epison: float = 1e-6):
        super().__init__()

        self.beta = nn.Parameter(torch.ones(dim))
        self.epison = epison
        
    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.to(torch.float32)
        mean_x_square = torch.pow(x, 2).mean(-1, keepdim=True)
        out = x.to(dtype) * torch.rsqrt(mean_x_square + self.epison) * self.beta.to(x.device)
        return out

class RoPE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

# Function to measure GPU memory
def measure_memory(device):
    return {
        "allocated": torch.cuda.memory_allocated(device) / (1024 ** 2),  # in MB
        "reserved": torch.cuda.memory_reserved(device) / (1024 ** 2),  # in MB
        "max_allocated": torch.cuda.max_memory_allocated(device) / (1024 ** 2),  # in MB
    }