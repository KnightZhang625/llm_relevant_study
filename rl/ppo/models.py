# coding:utf-8

import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
from pathlib import Path

class Actor(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim, lr, ckpt_dir="./saved_models/"):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(*input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.to(self.device)

        Path(ckpt_dir).mkdir(exist_ok=True)
        self.ckpt_path = Path(ckpt_dir) / "actor.pt"
    
    def forward(self, obs):
        """
            obs: [bsz, input_dim]
        """
        out = self.layers(obs)
        out = dist.categorical.Categorical(out)
        
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.ckpt_path)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.ckpt_path))

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, lr, ckpt_dir="./saved_models/"):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(*input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.to(self.device)

        Path(ckpt_dir).mkdir(exist_ok=True)
        self.ckpt_path = Path(ckpt_dir) / "critic.pt"
    
    def forward(self, obs):
        """
            obs: [bsz, input_dim]
        """
        value = self.layers(obs)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.ckpt_path)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.ckpt_path))