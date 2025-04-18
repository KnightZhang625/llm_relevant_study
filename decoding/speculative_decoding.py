# coding:utf-8

import torch
from torch.nn import functional as F

# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if (idx_next.item() == 0):
        raise RuntimeError
    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum

@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                            max_len : int , gamma : int = 4,
                            temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        prefix (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """

    T = prefix.size(1) + max_len

    while prefix.size(1) < T:
        prefix_len = prefix.size(1)
        x = prefix

        for _ in range(gamma):
            # logits: [b, x.size(1), V]
            q_logits = approx_model(x)
            next_tok = sample(norm_logits(q_logits[:, -1, :], temperature, top_k, top_p))
            x = torch.cat((x, next_tok), dim=1)
        
        # q_logits: [b, prefix_len + gamma, V]
        q_probs = torch.empty_like(q_logits)
        for i in range(q_logits.size(1)):
            q_probs[:, i, :] = norm_logits(q_logits[:, i, :], temperature, top_k, top_p)

        # p_logits: [b, prefix_len + gamma, V]
        p_logits = target_model(x)
        p_probs = torch.empty_like(p_logits)
        for i in range(p_logits.size(1)):
            p_probs[:, i, :] = norm_logits(p_logits[:, i, :], temperature, top_k, top_p)
        
        is_all_accept = True
        n = prefix_len - 1
        for i in range(gamma):
            r = torch.rand(1, device=prefix.device)
            j = x[:, prefix_len + i]
            ratio = p_probs[:, prefix_len + i - 1, j] / q_probs[:, prefix_len + i - 1, j]

            if r < ratio:
                n +=1
            else:
                t = sample(max_fn(p_probs[:, prefix_len + i - 1, n] - q_probs[:, prefix_len + i - 1, n])) 
        
        prefix = x[:, :n+1]
        if is_all_accept:
            t = sample(p_probs[:, -1, :])
        
        prefix = torch.cat((prefix, t), dim=-1)
    
    return prefix