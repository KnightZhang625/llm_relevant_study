# coding:utf-8

import copy
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

def print_trainable_parameters(model: nn.Module):
    all_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    percentage = trainable_params / all_params * 100
    print(f"All Params: {all_params:,} Trainable Params: {trainable_params:,} Percentage: {percentage:.2f}%")

@dataclass
class LoraConfig:

    lora_r: int = field(
        default=8,
        metadata={
            "help": "The rank for the matrix."
        }
    )

    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "The scaling factor."
        }
    )

    lora_dropout: float = field(
        default=0.0,
        metadata={
            "help": "Dropout rate for lora layer."
        }
    )

    lora_modules: list = field(
        default_factory=list,
        metadata={
            "help": "The layer to adopt lora."
        }
    )


class LoraLayer(nn.Module):

    def __init__(self,
                 module: nn.Module,
                 lora_r: int,
                 lora_alpha: int,
                 lora_dropout: float):
        super().__init__()
        
        self.ori_module = module
        self.weight_shape = module.weight.shape

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        self.lora_weight_A = nn.Parameter(
            torch.empty(
                (self.lora_r, self.weight_shape[1]), 
                dtype=module.weight.dtype, 
                device=module.weight.device,
                ), 
            requires_grad=True
        )
        self.lora_weight_B = nn.Parameter(
            torch.empty(
                (self.weight_shape[0], self.lora_r),
                dtype=module.weight.dtype,
                device=module.weight.device,
            ),
            requires_grad=True
        )

        nn.init.normal_(self.lora_weight_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_weight_B)

        self.dropout = nn.Dropout(lora_dropout)

        for param in self.ori_module.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.tensor) -> torch.Tensor:
        scale_factor = self.lora_alpha / self.lora_r
        out = torch.matmul(self.lora_weight_A, self.dropout(x.T))
        out = torch.matmul(self.lora_weight_B, out)
        return self.ori_module(x) + scale_factor * out

class Lora:

    def __init__(self, model: nn.Module, lora_config: LoraConfig):

        self.lora_r = lora_config.lora_r
        self.lora_alpha = lora_config.lora_alpha
        self.lora_dropout = lora_config.lora_dropout

        self.replace_lora_layer(model)

    def replace_lora_layer(self, 
                           module: nn.Module,
                           embed_requires_grad: bool=True,
                           norm_requires_grad: bool=True,
                           head_requires_grad: bool=True):

        for name, child in module.named_children():

            if "embed" in name:
                for param in child.parameters():
                    param.requires_grad = embed_requires_grad
            elif "norm" in name:
                for param in child.parameters():
                    param.requires_grad = norm_requires_grad
            elif "head" in name:
                for param in child.parameters():
                    param.requires_grad = head_requires_grad
            
            elif isinstance(child, nn.Linear):
                lora_layer = LoraLayer(
                    module=child,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                )
                setattr(module, name, lora_layer)
            
            else:
                self.replace_lora_layer(child, embed_requires_grad, norm_requires_grad, head_requires_grad)

def unload_lora(model: nn.Module, adapter_path: str="adapter.pt"):
    lora_parameters = {}

    def search_lora_layer(module: nn.Module, prefix: list[str]):
        for name, child in module.named_children():
            new_prefix = prefix + [name]
            if isinstance(child, LoraLayer):
                lora_parameters[".".join(new_prefix)] = {
                    "lora_weight_A": child.lora_weight_A.data.to("cpu"),
                    "lora_weight_B": child.lora_weight_B.data.to("cpu"),
                    "lora_r": child.lora_r,
                    "lora_alpha": child.lora_alpha,
                    "dropout": child.dropout.p,
                }
                setattr(module, name, child.ori_module)
            else:
                search_lora_layer(child, new_prefix)
    
    search_lora_layer(model, [])
        
    for param in model.parameters():
        param.requires_grad = True
    
    torch.save(lora_parameters, adapter_path)

def load_lora(model: nn.Module, adpter_path: str="adapter.pt"):
    lora_parameters = torch.load(adpter_path, weights_only=True)

    for lora_name, lora_param in lora_parameters.items():
        module = dict(model.named_modules())[lora_name]
        if isinstance(module, nn.Linear):
            lora_layer = LoraLayer(
                module=module,
                lora_r=lora_param["lora_r"],
                lora_alpha=lora_param["lora_alpha"],
                lora_dropout=lora_param["dropout"],
            )
            lora_layer.lora_weight_A.data = lora_param["lora_weight_A"]
            lora_layer.lora_weight_B.data = lora_param["lora_weight_B"]
            
            parts = lora_name.split(".")
            obj = model
            for attr_name in parts[:-1]:
                obj = getattr(obj, attr_name)
            setattr(obj, parts[-1], lora_layer)
    
    for name, param in model.named_parameters():
        if any(s in name for s in ["embed", "norm", "head"]):
            param.requires_grad = True

def merge_lora(model: nn.Module):

    def search_lora_layer(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                with torch.no_grad():
                    lora_adjustment = torch.matmul(child.lora_weight_B, child.lora_weight_A) * (child.lora_alpha / child.lora_r)
                    child.ori_module.weight.add_(lora_adjustment.to(model.device))
                    setattr(module, name, child.ori_module)
            else:
                search_lora_layer(child)
    
    search_lora_layer(model)
    for param in model.parameters():
        param.requires_grad = True

if __name__ == "__main__":
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    
    config = AutoConfig.for_model('llama')
    config.hidden_size = 24
    config.intermediate_size = config.hidden_size * 4
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.num_key_value_heads = 2
    config.vocab_size = 128

    raw_model = AutoModel.from_config(config).to(device)
    print_trainable_parameters(raw_model)

    lora_model = copy.deepcopy(raw_model)
    Lora(lora_model, lora_config=LoraConfig())
    print_trainable_parameters(lora_model)

    unload_lora(lora_model)
    print_trainable_parameters(lora_model)

    load_lora(lora_model)
    print_trainable_parameters(lora_model)

    merge_lora(lora_model)
    print_trainable_parameters(lora_model)