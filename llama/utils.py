# coding:utf-8

import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps  
        self.gamma = nn.Parameter(torch.ones(dim))  # learnable paramters
    
    def _norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean_square = torch.square(x).mean(dim=-1, keepdim=True)
        # reciprocal
        return x * torch.rsqrt(mean_square + self.eps)

    def forward(self, x: torch.tensor) -> torch.tensor:
        norm_x = self._norm(x.float()).type_as(x)
        return norm_x * self.gamma.to(x.device)

if __name__ == "__main__":
    device = torch.device("mps")
    x = torch.tensor([[10, 20]], dtype=torch.float32, device=device)

    rms_norm = RMSNorm(2)
    print(rms_norm(x))