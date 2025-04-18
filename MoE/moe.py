# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class MOEConfig:
    n_experts: int
    top_k: int

    hidden_size: int

    n_group: int = 8  # used by V2's Device-Limited Routing
    topk_group: int = 2


class BasicMOE(nn.Module):
    def __init__(self, feat_in: int, feat_out: int):
        super().__init__()
        self.FFN = nn.Linear(feat_in, feat_out)
    
    def forward(self, x: torch.FloatType):
        return self.FFN(x)

class SharedMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        
        self.shared_moe_layer = nn.ModuleList(
            [
                BasicMOE(config.hidden_size, config.hidden_size) for _ in range(config.n_experts)
            ]
        )

    def forward(self, x: torch.FloatTensor):
        """
            x: [bsz * seq_len, hsz]
        """
        
        # send x to each expert, each out is [bsz * seq_len, hsz]
        out = [expert(x) for expert in self.shared_moe_layer]
        out = torch.stack(out, dim=1)   # [bsz * seq_len, n_experts, hsz]
        return out.sum(dim=1)   # [bsz * seq_len, hsz]

class SparseMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()

        self.sparsed_moe_layer = nn.ModuleList(
            [
                BasicMOE(config.hidden_size, config.hidden_size) for _ in range(config.n_experts)
            ]
        )
        self.gate = nn.Linear(config.hidden_size, config.n_experts)
        self.n_experts = config.n_experts
        self.top_k = config.top_k

        self.expert_bias = nn.Parameter(torch.randn((config.n_experts)))
        self.n_group = config.n_group
        self.topk_group = config.topk_group
    
    def moe_router(self, x: torch.FloatTensor):
        """
            Input:
                - x: [bsz * seq_len, h]
            Return:
                - top_k_probs: [bsz * seq_len, top_k]
                - top_k_index: [bsz * seq_len, top_k]
                - mask: [n_experts, top_k, bsz * seq_len]
                - routing_probs: [bsz * seq_len, n_experts]
        """
        # [bsz * seq_len, n_experts]
        routing_logits = self.gate(x)
        routing_probs = F.softmax(routing_logits, dim=-1)
        # top_k_probs: [bsz * seq_len, top_k]
        # top_k_index: [bsz * seq_len, top_k]
        top_k_probs, top_k_index = torch.topk(routing_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=1, keepdim=True)
        top_k_probs = top_k_probs.to(x.dtype)

        # [bsz * seq_len, top_k, n_experts]: refer each token was selected by top_k experts
        mask = F.one_hot(top_k_index, num_classes=self.n_experts)
        # [n_experts, top_k, bsz * seq_len]: refer each expert, when was chosen as top_1, top_2, ..., top_n, which token were chosen for each top_n
        mask = mask.permute(2, 1, 0)

        return top_k_probs, top_k_index, mask, routing_probs

    def moe_router_v3(self, x: torch.FloatTensor):
        """
            Input:
                - x: [bsz * seq_len, h]
            Return:
                - topk_weight: [bsz * seq_len, top_k]
                - topk_idx: [bsz * seq_len, top_k]
                - mask: [n_experts, top_k, bsz * seq_len]
                - routing_probs: [bsz * seq_len, n_experts]
        """
        bsz_seq, h = x.shape

        # [bsz * seq_len, n_experts]
        logits = self.gate(x)
        # use sigmoid instead of softmax
        scores = F.sigmoid(logits)
        # [bsz * seq_len, n_experts] + [1, n_experts]
        scores_for_choices = scores + scores.unsqueeze(0)

        # group
        # [bsz * seq_len, n_group, n_experts / n_group]
        group_scores = scores_for_choices.view(bsz_seq, self.n_group, -1)
        # [bsz * seq_len, n_group]
        group_scores = torch.topk(group_scores, k=2, dim=-1)[0].sum(-1)
        # [bsz * seq_len, topk_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, scorted=False)[1]
        # [bsz * seq_len, n_group]
        group_mask = torch.zeros_like(group_scores)
        group_mask = group_mask.scatter_(1, group_idx, 1)

        # [bsz * seq_len, n_group] -> [bsz * seq_len, n_group, 1] -> [bsz * seq_len, n_group, n_experts / n_group] -> [bsz * seq_len, n_experts]
        scores_mask = group_mask.unsqueeze(-1).repeat(1, 1, 1, self.n_experts // self.n_group).view(bsz_seq, -1)

        tmp_scores = scores_for_choices.masked_fill_(~scores_mask.bool(), 0.0)
        # [bsz * seq_len, top_k]
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, scorted=False)
        topk_weight = scores.gather(1, topk_idx)

        # [bsz * seq_len, top_k, n_experts] -> [n_experts, top_k, bsz * seq_len]
        mask = F.one_hot(topk_idx, num_classes=self.n_experts).permute(2, 1, 0)

        # [bsz * seq_len, top_k]
        topk_weight_denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / topk_weight_denominator

        return topk_weight, topk_idx, mask, scores

    def forward(self, x: torch.FloatTensor):
        """
            x: [bsz, seq_len, hsz]
        """
        bsz, seq_len, hsz = x.shape
        x = x.view(-1, hsz)     # [bsz * seq_len, hsz]

        # top_k_probs: [bsz * seq_len, top_k]
        # top_k_index: [bsz * seq_len, top_k]
        # mask: [n_experts, top_k, bsz * seq_len]
        # routing_probs: [bsz * seq_len, n_experts]
        top_k_probs, top_k_index, mask, routing_probs = self.moe_router(x)
        final_output = torch.zeros_like(x)

        for idx in range(self.n_experts):
            expert_layer = self.sparsed_moe_layer[idx]

            # select the tokens chosen by this expert
            # as_top_k_idx: refers to which top_n assigned for this token
            # top_idx: the selected token index
            as_top_k_idx, tok_idx = torch.where(mask[idx])
       
            # selected_tokens_hidden: [n_selected_tokens, hsz]
            selected_tokens_hidden = x[tok_idx, :]
            # forward these tokens to this expert layer
            # selected_tokens_out: [n_selected_tokens, hsz]
            selected_tokens_out = expert_layer(selected_tokens_hidden)

            # multiply with the weights
            # selected_tokens_out: [n_selected_tokens, hsz] * [n_selected_tokens, 1]
            selected_tokens_out = selected_tokens_out * top_k_probs[tok_idx, as_top_k_idx].unsqueeze(-1)

            # add to final_output
            final_output.index_add_(0, tok_idx, selected_tokens_out)
        
        final_output = final_output.view(bsz, seq_len, hsz)

        return final_output, top_k_index, routing_probs

class MOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()

        self.shared_moe = SharedMOE(config)
        self.sparsed_moe = SparseMOE(config)
        self.config = config
        
    def cal_loss(self, x: torch.FloatTensor, target: torch.FloatTensor, top_k_index: torch.LongTensor, routing_probs: torch.FloatTensor):
        mse_loss = F.mse_loss(x, target)
        aux_loss = self.expert_level_balance_loss(top_k_index, routing_probs)
        return mse_loss + 0.01 * aux_loss, mse_loss, aux_loss
    
    def expert_level_balance_loss(self, top_k_index: torch.LongTensor, routing_probs: torch.FloatTensor):
        # top_k_index: [bsz * seq_len, top_k]
        # routing_probs: [bsz * seq_len, n_experts]

        # mask: [bsz * seq_len, top_k, n_experts]
        mask = F.one_hot(top_k_index, num_classes=self.config.n_experts)
        # mask: [bsz * seq_len * top_k, n_experts]
        mask = mask.view(-1, self.config.n_experts)
        # bsz * seq_len: all tokens
        # top_k: activate experts
        fi = mask.float().mean(dim=0) * self.config.n_experts   # [n_experts]

        pi = routing_probs.mean(dim=0)  # [n_experts]

        aux_loss = (fi * pi).sum()

        return aux_loss

    def forward(self, x: torch.FloatTensor, target: torch.FloatTensor = None):
        """
            x: [bsz, seq_len, hsz]
        """
        shared_out = self.shared_moe(x)
        spared_out, top_k_index, routing_probs = self.sparsed_moe(x)
        out = shared_out + spared_out

        loss, mse_loss, aux_loss = None, None, None
        if target is not None:
            loss, mse_loss, aux_loss = self.cal_loss(out, target, top_k_index, routing_probs)
        
        return out, loss, mse_loss, aux_loss

if __name__ == "__main__":
    config = MOEConfig(n_experts=8, top_k=4, hidden_size=32)
    bsz = 16
    seq_len = 10
    epochs = 1000

    moe_model = MOE(config)
    optimizer = torch.optim.Adam(moe_model.parameters(), lr=1e-3)
    moe_model.train()

    for e in range(epochs):
        x = torch.randn(bsz, seq_len, config.hidden_size)
        target = torch.randn(bsz, seq_len, config.hidden_size)

        out, loss, mse_loss, aux_loss = moe_model(x, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print(f"Epoch: {e}, Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, AUX: {aux_loss.item():.4f}")