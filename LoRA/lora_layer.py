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
                 test_mode:bool=False):
        
        super(LoraLinear, self).__init__()

        device, dtype = module.weight.data.device, module.weight.data.dtype
        out_feature, in_feature = module.weight.data.shape
        self.base_linear = copy.deepcopy(module)

        self.lora_a_weight = nn.Parameter(torch.empty((rank, in_feature), dtype=dtype, device=device))
        self.lora_b_weight = nn.Parameter(torch.empty((out_feature, rank), dtype=dtype, device=device))

        nn.init.normal_(self.lora_a_weight, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.lora_b_weight, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_b_weight)

        self.dropout = nn.Dropout(p=dropout)

        self.alpha = alpha
        self.rank = rank

        for param in self.base_linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        scale = self.alpha / self.rank
        # x: [bsz, in_features] * [in_features, rank] -> [bsz, rank]
        out = torch.matmul(self.dropout(x), self.lora_a_weight.T)
        # [bsz, rank] * [rank, out_features] -> [bsz, out_features]
        out = torch.matmul(out, self.lora_b_weight.T)
        return self.base_linear(x) + scale * out

def replace_linear_with_lora(model: nn.Module,
                             rank: int,
                             alpha: int,
                             dropout: float,
                             embed_requires_grad: bool=False,
                             norm_requires_grad: bool=False,
                             head_requires_grad: bool=False,
                             test_mode: bool=False,
                            ):
    
    for name, child in model.named_children():
        if any(n in name for n in ["embed", "norm", "lm_head"]):
            if "embed" in name:
                requires_grad = embed_requires_grad
            elif "norm" in name:
                requires_grad = norm_requires_grad
            elif "lm_head" in name:
                requires_grad = head_requires_grad

            for param in child.parameters():
                param.requires_grad = requires_grad
    
        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(
                module=child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                test_mode=test_mode,
            )
            setattr(model, name, lora_linear)
        
        else:
            replace_linear_with_lora(
                child,
                rank, alpha, dropout,
                embed_requires_grad, norm_requires_grad, head_requires_grad,
                test_mode
            )

def unload_lora(model: nn.Module, adapter_name: str="adapter"):
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
                setattr(module, name, child.base_linear)
            else:
                search_lora_layer(child, new_prefix)
        
    search_lora_layer(model, [])
    for param in model.parameters():
        param.requires_grad = True
    
    torch.save(lora_parameters, f"{adapter_name}.pt")

def load_lora(model: nn.Module, adapter_path: str):
    lora_parameters = torch.load(adapter_path)
    device = model.device

    for prefix, params in lora_parameters.items():
        child = dict(model.named_modules())[prefix]
        if isinstance(child, nn.Linear):
            lora_linear = LoraLinear(
                module = child,
                rank = params["rank"],
                alpha = params["alpha"],
                dropout = params["dropout"],
            )
            lora_linear.lora_a_weight.data = params["lora_a_weight"].to(device)
            lora_linear.lora_b_weight.data = params["lora_b_weight"].to(device)

            parts = prefix.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], lora_linear)
    
    for name, param in model.named_parameters():
        if any(s in name for s in ["embed", "norm", "lm_head"]):
            param.requires_grad = False

def merge_lora(model: nn.Module):
    
    def search_lora_linear(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, LoraLinear):
                lora_a_weight = child.lora_a_weight
                lora_b_weight = child.lora_b_weight
                with torch.no_grad():
                    lora_adjustment = torch.matmul(lora_b_weight, lora_a_weight)
                    child.base_linear.weight.add_(lora_adjustment)
                setattr(module, name, child.base_linear)
            else:
                search_lora_linear(child)
    
    search_lora_linear(model)
    for param in model.parameters():
        param.requires_grad = True

def print_trainable_parameters(model: nn.Module):
    total_params = sum([p.numel() for p in model.parameters()])
    trainalbe_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    trainable_percentage = trainalbe_params / total_params * 100

    print(f"Trainble Params: {trainalbe_params:,} || All Params: {total_params:,} || trainable%: {trainable_percentage:.2f}%")

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rank = 8
    alpha = 16
    dropout = 0.0

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
    replace_linear_with_lora(lora_model, rank, alpha, dropout)

    print_trainable_parameters(raw_model)
    print_trainable_parameters(lora_model)

    unload_lora(lora_model)
    print_trainable_parameters(lora_model)

    load_lora(lora_model, adapter_path="adapter.pt")
    print_trainable_parameters(lora_model)

    merge_lora(lora_model)
    print_trainable_parameters(lora_model)