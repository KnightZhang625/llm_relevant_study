# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEGate(nn.Module):
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape        
        ############################
        # 这里的hidden_states就是公式里的T，是一个Batch数据的全部token做计算，每个Batch会重新计算
        ############################
        hidden_states = hidden_states.view(-1, h)
        
        # logits: [bsz * seq_len, n_routed_experts]
        logits = F.linear(hidden_states, self.weight, None)
        # scores_for_aux: [bsz * seq_len, n_routed_experts]
        scores_for_aux = logits.softmax(dim=-1)
        
        # topk_idx: [bsz * seq_len, top_k]
        topk_weight, topk_idx = torch.topk(scores_for_aux, k=self.top_k, dim=-1, sorted=False)
        # topk_idx_for_aux_loss: [bsz, seq_len, top_k] -> [bsz, seq_len * top_k]
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        # mask_ce: [bsz * seq_len * top_k, n_routed_experts]
        mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
        # [n_routed_experts]
        # NOTE: 第一维是bsz*seq_len*top_k, bsz*seq_len=T(所有tokens), top_k=k'(激活专家数), 因此求mean已经就除过(K'T)了
        ce = mask_ce.float().mean(0)
        ############################
        # 计算Pi，fi 和 aux_loss。这里的计算并没有跨Batch累积，每个Batch单独计算
        ############################      
        Pi = scores_for_aux.mean(0)
        fi = ce * self.n_routed_experts
        aux_loss = (Pi * fi).sum() * self.alpha







