# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_moe import SparseMOE, MOEConfig
from basic_moe import BasicExpert

class SharedExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(
                    config.hidden_dim, config.hidden_dim
                ) for _ in range(config.shared_experts_number)
            ]
        )
    
    def forward(self, x):
        # sparse_moe_out: [b, s, h]
        # router_logits: [b*s, e]
        sparse_moe_out, router_logits = self.moe_model(x)

        # [b, s, h]
        shared_experts_output = [
            expert(x) for expert in self.shared_experts
        ]
        # [b, s, h]
        shared_experts_out = torch.stack(shared_experts_output, dim=0).sum(dim=0)

        return sparse_moe_out + shared_experts_out, router_logits

def expert_level_balance_loss(router_logits: torch.tensor, num_experts: int, top_k: int):
    # router_logits: [bsz * seq_len, num_experts]
    router_probs = F.softmax(router_logits, dim=-1)
    # topk_idx: [bsz * seq_len, top_k]
    _, topk_idx = torch.topk(router_logits, k=top_k, dim=-1)
    topk_idx = topk_idx.view(-1)    # [bsz * seq_len * top_k]
    mask = F.one_hot(topk_idx, num_classes=num_experts) # [bsz * seq_len * top_k, num_experts]
    # bsz*seq_len: all tokens
    # top_k: activated experts number
    # bsz*seq_len*top_k=TK'
    fi = mask.float().mean(0) * num_experts # [num_experts]
    pi = router_probs.mean(0)       # [num_experts]
    aux_loss = (fi * pi).sum()
    
    return aux_loss

def test_moe_training():
    batch_size = 32
    seq_len = 16
    hidden_dim = 32
    num_batches = 1000

    config = MOEConfig(
        hidden_dim=hidden_dim,
        expert_number=4,
        top_k=2,
        shared_experts_number=2,
    )

    model = SharedExpertMOE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for batch in range(num_batches):
        x = torch.randn(batch_size, seq_len, hidden_dim)
        target = torch.randn(batch_size, seq_len, hidden_dim)

        output, router_logits = model(x)

        mse_loss = F.mse_loss(output, target)
        aux_loss = expert_level_balance_loss(router_logits, config.expert_number, config.top_k)

        total_loss = mse_loss + 0.01 * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"Batch {batch}, Loss: {total_loss.item():.4f}, MSE: {mse_loss.item():.4f}, Aux: {aux_loss.item():.4f}")

if __name__ == "__main__":
    test_moe_training()