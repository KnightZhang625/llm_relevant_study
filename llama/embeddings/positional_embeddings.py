# coding:utf-8

import torch
import torch.nn as nn

def sinusoidal_embedding(max_length: int=512, dim: int=1024):
    row_vector = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    col_vector = torch.arange(0, dim // 2, dtype=torch.float32).unsqueeze(0)

    row_col_dot = row_vector @ (1 / torch.pow(10000.0, 2 * col_vector / dim))
    
    out_vector = torch.empty(max_length, dim, dtype=torch.float32)
    out_vector[:, 0::2] = torch.sin(row_col_dot)
    out_vector[:, 1::2] = torch.cos(row_col_dot)

    return out_vector

class RoPEPositionalEmbedding:
    
    def __init__(self, max_seq_len: int, dim: int):
        self.rotation_matrix = self.precompute_rotation_matrix(max_seq_len, dim)

    def precompute_rotation_matrix(self, max_seq_len, dim):
        theta = 1 / (10000 ** (torch.arange(0, dim // 2) * 2 / dim))
        theta = theta.unsqueeze(0)  # (1, dim / 2)
        
        pos_idx = torch.arange(0, max_seq_len).unsqueeze(1) # (max_seq_len, 1)

        m_theta = pos_idx * theta
        
        # torch.polar(abs, angle), cos(m*theta) + i * sin(m*theta)
        rotation_matrix = torch.polar(torch.ones_like(m_theta), m_theta)

        return rotation_matrix
    
    def apply_rotary_emb(self, q, k):
        # q: (b, n_head, s, d) -> qx_: (b, n_head, s, d//2, 2)
        qx_ = torch.reshape(q, (*q.shape[:-1], -1, 2))
        kx_ = torch.reshape(k, (*k.shape[:-1], -1, 2))

        # qx_: (b, n_head, s, d//2, 2) -> qx_complex: (b, n_head, s, d//2)
        qx_complex = torch.view_as_complex(qx_)
        kx_complex = torch.view_as_complex(kx_)

        # (b, n_head, s, d//2) * (s, d//2) -> (b, n_head, s, d//2) -> (b, n_head, s, d//2, 2) -> (b, n_head, s, d)
        q_out = torch.view_as_real(qx_complex * self.rotation_matrix).flatten(start_dim=-2)
        k_out = torch.view_as_real(kx_complex * self.rotation_matrix).flatten(start_dim=-2)

        return q_out, k_out

if __name__ == "__main__":
    # embeddings = sinusoidal_embedding(max_length=8, dim=4)
    # print(embeddings)

    rope = RoPEPositionalEmbedding(8, 4)
    q = torch.rand((2, 6, 8, 4))
    k = torch.rand((2, 6, 8, 4))
    rope.apply_rotary_emb(q, k)