# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_moe import BasicExpert

class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k

    def forward(self, hidden_states):
        # [b, s, e]
        router_logits = self.gate(hidden_states)
        # [b, s, e]
        routing_probs = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_idx = torch.topk(
            routing_probs, self.top_k, dim=-1
        )   # [b, s, top_k]

        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        expert_mask = F.one_hot(
            top_k_idx,
            num_classes=self.expert_number,
        )   # [b, s, top_k, e]

        # router_logits: [b, s, e]
        # top_k_weights: [b, s, top_k]
        # top_k_idx: [b, s, top_k]
        # expert_mask: [b, s, top_k, e]
        return router_logits, top_k_weights, top_k_idx, expert_mask

class MOEConfig:
    def __init__(
            self,
            hidden_dim,
            expert_number,
            top_k,
            shared_experts_number=2,
    ):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number

class SparseMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number
        self.top_k = config.top_k
        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )
        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)

    def forward(self, x):
        bsz, seq_len, hsz = x.size()

        # hidden_states: [bsz*seq_len, hsz]
        hidden_states = x.view(-1, hsz)

        # router_logits: [b, s, e]
        # top_k_weights: [b, s, top_k]
        # top_k_idx: [b, s, top_k]
        # expert_mask: [b, s, top_k, e]
        router_logits, top_k_weights, top_k_idx, expert_mask = self.router(hidden_states)
        router_logits = router_logits.view(-1, self.expert_number)  # [b*s, e]
        top_k_weights = top_k_weights.view(-1, self.top_k)  # [b*s, topk]
        top_k_idx = top_k_idx.view(-1, self.top_k)  # [b*s, topk]
        expert_mask = expert_mask.view(-1, self.top_k, self.expert_number)  # [b*s, topk, e]
        # 每个expert, 被选作topk, top0, top1, top2, ..., 都有哪些token
        # 假设topk=2, 下面这个expert被选作top0的token有2,4, top1的token有1
        # [0, 0, 1, 0, 1]
        # [0, 1, 0, 0, 0]
        expert_mask = expert_mask.permute(2, 1, 0)  # [e, top_k, b*s]

        final_hidden_states = torch.zeros(
            (bsz * seq_len, hsz),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # idx: range(0 -> topk),
            # top_x: range(vocab_size)
            # e.g. idx[0, 0, 1], top_x[2, 4, 1], 代表top0有2,4, top1有1
            idx, top_x = torch.where(expert_mask[expert_idx])

            # hidden_states: [b*s, h] -> [1, top_x, h] -> [top_x, h]
            # 这一步选取这个expert选到的tokens
            current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(-1, hsz)
   
            # top_k_weights[b*s, topk]: 每个token对于所有expert选择的分数的topk
            # top_k_weights[top_x, idx]: top_x: 当前expert选择的token, idx: 和这些token选择这个expert最为top几?
            # expert_layer(current_state): [top_x, h] 每个token的在这个expert下的输出, 乘上这个token选择这个expert为top几的权重
            current_hidden_state = expert_layer(current_state) * top_k_weights[top_x, idx].unsqueeze(-1)

            final_hidden_states.index_add_(0, top_x, current_hidden_state.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.view(bsz, seq_len, hsz)

        return final_hidden_states, router_logits
    
if __name__ == "__main__":
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 4, 2)
    token_level_moe = SparseMOE(config)
    out, router = token_level_moe(x)
    print(out.shape)