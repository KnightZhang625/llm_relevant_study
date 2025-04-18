# coding:utf-8

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoraLinear(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 rank: int,
                 alpha: int,
                 dropout: float,
                 test_mode: bool = False):

        super(LoraLinear, self).__init__()

        device = module.weight.device
        dtype = module.weight.dtype
        in_features = module.weight.data.shape[1]
        out_features = module.weight.data.shape[0]
        self.ori_layer = copy.deepcopy(module)

        self.lora_a_weight = nn.Parameter(torch.empty([rank, in_features], device=device, dtype=dtype))
        self.lora_b_weight = nn.Parameter(torch.empty([out_features, rank], device=device, dtype=dtype))
        self.rank = rank
        self.alpha = alpha

        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.lora_a_weight, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.lora_b_weight, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_b_weight)
        
        for param in self.ori_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.tensor):
        scale = self.alpha / self.rank
        # [/, in_features] * [in_features, rank] -> [/, rank]
        out = torch.matmul(self.dropout(x), self.lora_a_weight.T)
        # [/, rank] * [rank, out_features] -> [/, out_feature]
        out = torch.matmul(out, self.lora_b_weight.T)
        return self.ori_layer(x) + scale * out

def replace_linear_with_lora(
        module: nn.Module,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        embed_requires_grad: bool = False,
        norm_requires_grad: bool = False,
        head_required_grad: bool = False,
        test_mode: bool = False,
):
    for name, child in module.named_children():
        if any(s in name for s in ["embed", "norm", "lm_head"]):
            if "embed" in name:
                requires_grad = embed_requires_grad
            elif "norm" in name:
                requires_grad = norm_requires_grad
            elif "lm_head" in name:
                requires_grad = head_required_grad
            
            for param in child.parameters():
                param.requires_grad = requires_grad
        
        elif isinstance(child, nn.Linear):
            lora_layer = LoraLinear(
                module=child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                test_mode=test_mode,
            )
            setattr(module, name, lora_layer)
        
        else:
            replace_linear_with_lora(
                module=child,
                rank=rank, alpha=alpha, dropout=dropout,
                embed_requires_grad=embed_requires_grad,
                norm_requires_grad=norm_requires_grad,
                head_required_grad=head_required_grad,
                test_mode=test_mode,
            )

def unload_lora(module: nn.Module, adapter_name: str="adapter"):
    lora_parameters = {}

    def search_lora_layer(module: nn.Module, prefix: list[str]):
        for name, child in module.named_children():
            new_prefix = prefix + [name]
            if isinstance(child, LoraLinear):
                lora_parameters[".".join(new_prefix)] = {
                    "lora_a_weight": child.lora_a_weight.data.cpu(),
                    "lora_b_weight": child.lora_b_weight.data.cpu(),
                    "rank": child.rank,
                    "alpha": child.alpha,
                    "dropout": child.dropout.p,
                }       
                setattr(module, name, child.ori_layer)
            else:
                search_lora_layer(child, new_prefix)
    
    search_lora_layer(module, [])
    for param in module.parameters():
        param.requires_grad = True
    
    torch.save(lora_parameters, f"{adapter_name}.pt")

def load_lora(model: nn.Module, adapter_path: str):
    lora_parameters = torch.load(adapter_path)
    device = model.device

    for lora_name, lora_param in lora_parameters.items():
        child = dict(model.named_modules())[lora_name]
        if isinstance(child, nn.Linear):
            lora_layer = LoraLinear(
                module=child,
                rank=lora_param["rank"],
                alpha=lora_param["alpha"],
                dropout=lora_param["dropout"],
            )
            lora_layer.lora_a_weight.data = lora_param["lora_a_weight"].to(device)
            lora_layer.lora_b_weight.data = lora_param["lora_b_weight"].to(device)

            parts = lora_name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], lora_layer)
    
    for name, param in model.named_parameters():
        if any(s in name for s in ["embed", "norm", "lm_head"]):
            param.requires_grad = False

def merge_lora(model: nn.Module):

    def search_lora_layer(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, LoraLinear):
                with torch.no_grad():
                    lora_adjustments = torch.matmul(child.lora_b_weight.data, child.lora_a_weight.data) * (child.alpha / child.rank)
                    child.ori_layer.weight.add_(lora_adjustments)
                
                setattr(module, name, child.ori_layer)

            else:
                search_lora_layer(child)
    
    search_lora_layer(model)
    for param in model.parameters():
        param.requires_grad = True


def print_trainable_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params

    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.2f}%")

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    
    config = AutoConfig.for_model('llama')
    config.hidden_size = 24
    config.intermediate_size = config.hidden_size * 4
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.num_key_value_heads = 2
    config.vocab_size = 128

    raw_model = AutoModel.from_config(config).to(device)
    lora_model = copy.deepcopy(raw_model).to(device)
    replace_linear_with_lora(lora_model)

    print_trainable_parameters(raw_model)
    print_trainable_parameters(lora_model)

    unload_lora(lora_model)
    print_trainable_parameters(lora_model)

    load_lora(lora_model, "adapter.pt")
    print_trainable_parameters(lora_model)

    merge_lora(lora_model)
    print_trainable_parameters(lora_model)