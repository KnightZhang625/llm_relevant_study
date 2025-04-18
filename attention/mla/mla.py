# coding:utf-8

import time
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm

from utils import RoPE, RMSNorm, measure_memory

@dataclass
class MLAConfig:
    # main
    hidden_size: int = 5120
    num_heads: int = 128
    attn_dropout: float = 0.0
    bias: bool = False

    # q
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64

    # k, v
    kv_lora_rank: int = 512
    v_head_dim: int = 128

class MLA(nn.Module):

    def __init__(self, config: MLAConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.attn_dropout = config.attn_dropout
        self.bias = config.bias

        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim

        # Query
        self.q_compress_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=self.bias,
        )   # W_{DQ} in (37)
        self.q_decompress_proj = nn.Linear(
            self.q_lora_rank,
            (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_heads,
            bias=self.bias,
        )   # W_{UQ}, W_{QR} in (38), (39)
        self.q_rms_norm = RMSNorm(dim=self.q_lora_rank)

        # Key, Value
        self.kv_compress_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=self.bias,
        )   # W_{DKV}, W_{KR} in (41), (43)
        self.kv_decompress_proj = nn.Linear(
            self.kv_lora_rank,
            (self.qk_nope_head_dim + self.v_head_dim) * self.num_heads,
            bias=self.bias
        )   # W_{UK}, W_{UV} in (42), (44)
        self.kv_rms_norm = RMSNorm(dim=self.kv_lora_rank)

        # Value to O
        self.o_proj = nn.Linear(
            self.v_head_dim * self.num_heads,
            self.hidden_size,
            bias=self.bias,
        )

        self.dropout = nn.Dropout(p=self.attn_dropout)
        self.rope = RoPE()  # dummy RoPE

    def forward(self, hidden_states: torch.Tensor, compressed_kv_cache: torch.Tensor = None):
        """Implement basic MLA, without matrix absorption.

        Args:
            - hidden_states: [bsz, q_len, hsz]
            - compressed_kv_cache: [bsz, prev_kv_len, kv_lora_rank + qk_rope_head_dim]
        """

        bsz, q_len, _ = hidden_states.size()

        """Query: compress -> decompress"""
        # [bsz, q_len, hsz] ->(q_compress_proj)-> [bsz, q_len, q_lora_rank]
        # [bsz, q_len, q_lora_rank] ->(kv_decompress_proj)-> [bsz, q_len, (qk_nope_head_dim + qk_rope_head_dim) * num_heads]
        q_nope_rope_comb = self.q_decompress_proj(self.q_rms_norm(self.q_compress_proj(hidden_states)))
        # q_nope: [bsz, q_len, num_heads, qk_nope_head_dim]
        # q_rope: [bsz, q_len, num_heads, qk_rope_head_dim]
        q_nope, q_rope = torch.split(
            q_nope_rope_comb.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim).transpose(1, 2),
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        """Key, Value: compress, concatenate with cache"""
        # compress_kv: [bsz, q_len, kv_lora_rank + qk_rope_head_dim]
        compress_kv = self.kv_compress_proj(hidden_states)
        p_len = 0
        if compressed_kv_cache != None:
            p_len = compressed_kv_cache.size(1)
            # [bsz, prev_kv_len(p_len) + q_len, kv_lora_rank + qk_rope_head_dim]
            compress_kv = torch.cat(
                (compressed_kv_cache, compress_kv),
                dim=1,
            )
        compressed_kv_cache = compress_kv.detach()
        
        # compress_kv: [bsz, (p_len) + q_len, kv_lora_rank]
        # k_rope: [bsz, (p_len) + q_len, qk_rope_head_dim]
        compress_kv, k_rope = torch.split(compress_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # [bsz, (p_len) + q_len, qk_rope_head_dim] -> [bsz, 1, (p_len) + q_len, qk_rope_head_dim]
        k_rope = k_rope.unsqueeze(2).transpose(1, 2)
        
        """Key, Value"""
        # kv: [bsz, (p_len) + q_len, (qk_nope_head_dim + v_head_dim) * num_heads]
        kv = self.kv_decompress_proj(self.kv_rms_norm(compress_kv))
        # k_nope: [bsz, num_heads, (p_len) + q_len, qk_nope_head_dim]
        # v: [bsz, num_heads, (p_len) + q_len, v_head_dim]
        k_nope, v = torch.split(
            kv.view(bsz, p_len + q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2),
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1
        )

        """q_rope, k_rope"""
        q_rope = self.rope(q_rope)
        k_rope = self.rope(k_rope)

        """Concatenate [q_nope, q_rope], [k_nope, k_rope]"""
        q = torch.empty((bsz, self.num_heads, q_len, self.qk_nope_head_dim + self.qk_rope_head_dim)).to(hidden_states.device)
        k = torch.empty((bsz, self.num_heads, p_len + q_len, self.qk_nope_head_dim + self.qk_rope_head_dim)).to(hidden_states.device)
        q[..., :self.qk_nope_head_dim] = q_nope
        q[..., self.qk_nope_head_dim:] = q_rope
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_rope

        """Attention Weights"""
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        
        """Attention Probs"""
        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        # [bsz, num_heads, q_len, (p_len) + q_len]
        attn_probs = self.dropout(attn_probs)

        """Output"""
        # [bsz, num_heads, q_len, (p_len) + q_len] @ [bsz, num_heads, (p_len) + q_len, v_head_dim] -> [bsz, num_heads, q_len, v_head_dim]
        output = attn_probs @ v
        output = torch.reshape(
            output.transpose(1, 2).contiguous(),
            (bsz, q_len, self.num_heads * self.v_head_dim)
        )
        # [bsz, q_len, num_heads * v_head_dim] -> [bsz, q_len, hsz]
        output = self.o_proj(output)

        return output, compressed_kv_cache
    
    def forward_a_cc_me(self, hidden_states: torch.Tensor, compressed_kv_cache: torch.Tensor = None):
        """Implement optimized MLA, absorb cache compressed.

        Args:
            - hidden_states: [bsz, q_len, hsz]
            - compressed_kv_cache: [bsz, prev_kv_len, kv_lora_rank + qk_rope_head_dim]
        """

        bsz, q_len, _ = hidden_states.size()

        """Query: compress -> decompress"""
        # [bsz, q_len, hsz] ->(q_compress_proj)-> [bsz, q_len, q_lora_rank]
        # [bsz, q_len, q_lora_rank] ->(kv_decompress_proj)-> [bsz, q_len, (qk_nope_head_dim + qk_rope_head_dim) * num_heads]
        q_nope_rope_comb = self.q_decompress_proj(self.q_rms_norm(self.q_compress_proj(hidden_states)))
        # q_nope: [bsz, num_heads, q_len, qk_nope_head_dim]
        # q_rope: [bsz, num_heads, q_len, qk_rope_head_dim]
        q_nope, q_rope = torch.split(
            q_nope_rope_comb.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim).transpose(1, 2),
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        """Absorb Weight"""
        # [num_heads, (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        kv_decompress_weight = self.kv_decompress_proj.weight.view(self.num_heads, (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank)
        # q_absorb: [num_heads, qk_nope_head_dim, kv_lora_rank]
        q_absorb = kv_decompress_weight[:, :self.qk_nope_head_dim, :]
        # o_absorb: [num_heads, v_head_dim, kv_lora_rank]
        o_absorb = kv_decompress_weight[:, self.qk_nope_head_dim:, :]

        """Key, Value: compress"""
        # compress_kv: [bsz, q_len, kv_lora_rank + qk_rope_head]
        compress_kv = self.kv_compress_proj(hidden_states)
        p_len = 0
        if compressed_kv_cache != None:
            p_len = compressed_kv_cache.size(1)
            # [bsz, p_len + q_len, kv_lora_rank + qk_rope_head]
            compress_kv = torch.cat((compressed_kv_cache, compress_kv), dim=1)
        compressed_kv_cache = compress_kv.detach()

        # compress_kv: [bsz, (p_len) + q_len, kv_lora_rank]
        # k_rope: [bsz, (p_len) + q_len, qk_rope_head_dim]
        compress_kv, k_rope = torch.split(
            compress_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        # k_rope: [bsz, 1, (p_len) + q_len, qk_rope_head]
        k_rope = k_rope.unsqueeze(2).transpose(1, 2)
        # compress_kv: [bsz, 1, (p_len) + q_len, kv_lora_rank]
        compress_kv = self.kv_rms_norm(compress_kv).unsqueeze(2).transpose(1, 2)

        q_rope = self.rope(q_rope)
        k_rope = self.rope(k_rope)

        """Attention Weights"""
        # [bsz, num_heads, q_len, qk_nope_head_dim] @  [num_heads, qk_nope_head_dim, kv_lora_rank] -> [bsz, num_heads, q_len, kv_lora_rank]
        q_nope = q_nope @ q_absorb
        # [bsz, num_heads, q_len, kv_lora_rank] @ [bsz, 1, kv_lora_rank, (p_len) + q_len] -> [bsz, num_heads, q_len, (p_len) + q_len]
        # [bsz, num_heads, q_len, qk_rope_head_dim] @ [bsz, 1, qk_rope_head_dim, (p_len) + q_len] -> [bsz, num_heads, q_len, (p_len) + q_len]
        attn_weights = (torch.matmul(q_nope, compress_kv.transpose(2, 3)) + torch.matmul(q_rope, k_rope.transpose(2, 3))) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        
        """Attention Probs"""
        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_nope.dtype)
        attn_probs = self.dropout(attn_probs)

        """Output"""
        # [bsz, num_heads, q_len, (p_len) + q_len] @ [bsz, 1, (p_len) + q_len, kv_lora_rank] -> [bsz, num_heads, q_len, kv_lora_rank]
        output = attn_probs @ compress_kv
        # [bsz, num_heads, q_len, kv_lora_rank] @ [1, num_heads,  kv_lora_rank, v_head_dim] -> [bsz, num_heads, q_len, v_head_dim]
        output = torch.matmul(output, o_absorb.unsqueeze(0).transpose(2, 3))
        output = output.transpose(1, 2).contiguous()
        output = torch.reshape(output, (bsz, q_len, self.num_heads * self.v_head_dim))
        
        output = self.o_proj(output)

        return output, compressed_kv_cache

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    config = MLAConfig()
    mla = MLA(config).to(device)

    pre_x = torch.randn(2, 128, 5120).to(device)
    follow_x = torch.randn(2, 100, 5120).to(device)

    """1. no cache"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    s = time.time()
    x = pre_x
    for idx in tqdm(range(follow_x.size(1))):
        attn_output, _ = mla(x)
        next_token = follow_x[:, idx, :].unsqueeze(1)
        x = torch.cat((x, next_token), dim=1)
    e = time.time()
    memory_forward = measure_memory(device)
    print(f"Use: {e - s:.2f}s")
    print(f"Forward Pass Memory Usage: {memory_forward}")
    case_1_output = attn_output[:, -1, :].unsqueeze(1)

    """2. cache compress_kv"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    compress_kv = None
    s = time.time()
    x = pre_x
    for idx in tqdm(range(follow_x.size(1))):
        attn_output, compress_kv = mla(x, compress_kv)
        x = follow_x[:, idx, :].unsqueeze(1)
    e = time.time()
    memory_forward = measure_memory(device)
    print(f"Use: {e - s:.2f}s")
    print(f"Forward Pass Memory Usage: {memory_forward}")
    case_2_output = attn_output

    """3. a_cc_me"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    compress_kv = None
    s = time.time()
    x = pre_x
    for idx in tqdm(range(follow_x.size(1))):
        attn_output, compress_kv = mla.forward_a_cc_me(x, compress_kv)
        x = follow_x[:, idx, :].unsqueeze(1)
    e = time.time()
    memory_forward = measure_memory(device)
    print(f"Use: {e - s:.2f}s")
    print(f"Forward Pass Memory Usage: {memory_forward}")
    case_3_output = attn_output

    if torch.allclose(case_1_output, case_2_output, rtol=1e-02, atol=1e-02):
        print("no cache and cache are identical.")
    if torch.allclose(case_2_output, case_3_output, rtol=1e-02, atol=1e-02):
        print("cache and a_cc_me are identical.")