# coding:utf-8

import torch
import torch.nn as nn

class BasicExpert(nn.Module):
    def __init__(self, feature_in: int, feature_out: int):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)
    
    def forward(self, x: torch.tensor):
        return self.linear(x)

class BasicMOE(nn.Module):
    def __init__(self, feature_in: int, feature_out: int, expert_number: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                BasicExpert(feature_in, feature_out) for _ in range(expert_number)
            ]
        )
        self.gate = nn.Linear(feature_in, expert_number)
    
    def forward(self, x):
        # x: [b, h]
        expert_weight = self.gate(x)    # [b, e]
        expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ]   # each: [b, 1, h]

        expert_output = torch.cat(expert_out_list, dim=1) # [b, e, h], each data needs e experts output, in total, b*e
        expert_weight = expert_weight.unsqueeze(1)  # [b, 1, e]
        output = torch.matmul(expert_weight, expert_output).squeeze(1) # [b, 1, h] -> [b, h]

        return output

if __name__ == "__main__":
    x = torch.rand(2, 8)

    basic_moe = BasicMOE(8, 6, 10)

    output = basic_moe(x)

    print(output.shape)