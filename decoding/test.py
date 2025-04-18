# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

def top_k_filter(logits: torch.Tensor, k: int) -> torch.FloatTensor:
    # logits: [b, V]
    top_k_logits = torch.topk(logits, k=k)[0]
    # top_k_logits[:, [-1]]: select the smallest of the top_k_logits,
    # and keep dim [b, 1] instead of [b]
    logits[logits < top_k_logits[:, [-1]]] = float("-inf")
    return logits

def top_p_filter(logits: torch.Tensor, p: float) -> torch.FloatTensor:
    # probs: [b, V]
    sorted_probs, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_probs, dim=-1), dim=-1)
    to_remove_indices = cum_probs > p
    # shift one step to the right, and make the first one be 0, promising at least one token is selected
    # [False, False, True, True, True] -> [False, False, False, True, True]
    to_remove_indices[..., 1:] = to_remove_indices[..., :-1]
    to_remove_indices[..., 0] = 0   # [b, V]
    to_remove_indices_back_ori_indices = to_remove_indices.scatter(1, sorted_indices, to_remove_indices)
    logits[to_remove_indices_back_ori_indices] = float("-inf")
    return logits

def norm_logits(logits: torch.tensor, temperature: float=0.0, top_k: int=0, top_p: float=0.0):
    # logits: [b, V]
    logits = logits / temperature
    if top_k != 0:
        logits = top_k_filter(logits)
    elif top_p != 0.0:
        pass
    return F.softmax(logits, dim=-1)

def sample(probs):
    # probs: [b, v]
    return torch.multinomial(probs, num_samples=1)

def max_fn(logits_diff):
    # logits_diff: [b, V]
    logits = torch.where(logits_diff > 0, logits_diff, torch.zeros_like(logits_diff))
    logits_sum = torch.sum(logits, dim=-1, keepdim=True)
    return logits / logits_sum

@torch.no_grad()
def speculative_decoding(prefix: torch.LongTensor, 
                         small_llm: nn.Module, 
                         target_llm: nn.Module, 
                         max_len: int, 
                         gamma: int):
    while len(prefix) < max_len:
        prefix_len = prefix.size(1)
        x = prefix

        for _ in range(gamma):
            small_logits = small_llm(x)
            pred_token_idx = sample(norm_logits(small_logits))
            x = torch.cat((x, pred_token_idx), dim=-1)  # [b, s]
        
        for idx in range(small_logits.size(1)):
            small_logits[:, idx, :] = norm_logits(small_logits[:, idx, :])
        
        # [b, prefix_len + gamma, V]
        target_logits = target_llm(x)
        for idx in range(target_logits.size(1)):
            target_logits[:, idx, :] = norm_logits(target_logits[:, idx, :])

        is_all_accept = True
        last_accept_idx = prefix_len - 1
        for idx in range(gamma):
            random_p = torch.rand(1)
            token_idx = x[:, prefix_len + idx]
            ratio = target_logits[:, last_accept_idx, token_idx] / small_logits[:, last_accept_idx, token_idx]
            if random_p < torch.min(torch.tensor([1]), ratio):
                last_accept_idx +=1
            else:
                last_token_idx = sample(max_fn(target_logits[:, last_accept_idx, :] - small_logits[:, last_accept_idx, :]))
                is_all_accept = False
                break
        
        prefix = x[:, :last_accept_idx + 1]

        if is_all_accept:
            last_token_idx = sample(target_logits[:, -1, :])
        
        prefix = torch.cat((prefix, last_token_idx), dim=-1)

        return prefix



