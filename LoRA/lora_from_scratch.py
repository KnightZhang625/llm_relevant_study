# coding:utf-8

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int=8, alpha: int=16, dropout_p: float=0.0, test_mode: bool=False):
        super(LoraLinear, self).__init__()

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.base_layer = copy.deepcopy(base_layer)
        self.dropout_p = nn.Dropout(dropout_p)

        self.r = r
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.empty((r, in_features), dtype=dtype, device=device))
        self.lora_b = nn.Parameter(torch.empty((out_features, r), dtype=dtype, device=device))

        nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.lora_b, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_b)
        
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        scale = self.alpha / self.r
        out = F.linear(self.dropout_p(x), self.lora_a)
        out = F.linear(out, self.lora_b)
        return self.base_layer(x) + scale * out
    
def replace_linear_with_lora(
        module: nn.Module,
        r: int=8,
        alpha: int=16,
        dropout_p: float=0.0,
        embed_requires_grad: bool = False,
        norm_requires_grad: bool = False,
        head_requires_grad: bool = False,
        test_mode: bool = False,
):
    for name, child in module.named_children():
        if any(s in name for s in ["embed", "norm", "lm_head"]):
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
                child,
                r, alpha, dropout_p,
                test_mode)
            setattr(module, name, lora_linear)
        
        else:
            replace_linear_with_lora(
                child,
                r, alpha, dropout_p,
                embed_requires_grad, norm_requires_grad, head_requires_grad,
                test_mode)

def print_trainable_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params

    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.2f}%")

def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name:50} | Requires_grad: {param.requires_grad}")

def unload_lora(module: nn.Module, adapter_name: str="adapter"):
    lora_parameters = {}

    def search_lora_linear(module: nn.Module, prefix: list[str]):
        for name, child in module.named_children():
            new_prefix = prefix + [name]
            if isinstance(child, LoraLinear):
                lora_parameters[".".join(new_prefix)] = {
                    "lora_A_weight": child.lora_a.data.cpu(),
                    "lora_B_weight": child.lora_b.data.cpu(),
                    "r": child.r,
                    "alpha": child.alpha,
                    "dropout_p": child.dropout_p.p,
                }
                setattr(module, name, child.base_layer)
            else:
                search_lora_linear(child, new_prefix)
        
    search_lora_linear(module, [])
    for param in module.parameters():
        param.requires_grad = True
    
    torch.save(lora_parameters, f"{adapter_name}.pt")

def load_lora(model: nn.Module, adapter_path: str):
    lora_parameters = torch.load(adapter_path)
    device = model.device

    for name, lora_params in lora_parameters.items():
        # named_modules(): return name, module
        child = dict(model.named_modules())[name]
        if isinstance(child, nn.Linear):
            lora_linear = LoraLinear(
                base_layer=child,
                r=lora_params["r"],
                alpha=lora_params["alpha"],
                dropout_p=lora_params["dropout_p"],
            )
            lora_linear.lora_a.data = lora_params["lora_A_weight"].to(device)
            lora_linear.lora_b.data = lora_params["lora_B_weight"].to(device)

            parts = name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], lora_linear)

    for name, params in model.named_parameters():
        if any(s in name for s in ["embed", "norm", "lm_head"]):
            params.requires_grad = False       

if __name__ == "__main__":

    device = "cpu" if not torch.backends.mps.is_available() else "mps"
    
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    config = AutoConfig.for_model('llama')
    config.hidden_size = 24
    config.intermediate_size = config.hidden_size * 4
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.num_key_value_heads = 2
    config.vocab_size = 128

    raw_model = AutoModel.from_config(config).to(device)

    test_tensor = torch.randint(0, config.vocab_size, (2, 8)).to(device)
    lora_model = copy.deepcopy(raw_model)
    replace_linear_with_lora(lora_model, test_mode=True)

    raw_model.eval()
    raw_res = raw_model(test_tensor).last_hidden_state
    lora_model.eval()
    before_unload_res = lora_model(test_tensor).last_hidden_state

    print_trainable_parameters(raw_model)
    print_trainable_parameters(lora_model)

    unload_lora(lora_model)
    lora_model.eval()
    unload_res = lora_model(test_tensor).last_hidden_state
    print_trainable_parameters(lora_model)

    load_lora(lora_model, "adapter.pt")
    lora_model.eval()
    load_res = lora_model(test_tensor).last_hidden_state
    print_trainable_parameters(lora_model)
    
    print(torch.allclose(raw_res, unload_res, atol=1e-6))
    print(torch.allclose(before_unload_res, load_res, atol=1e-6))
    print(torch.allclose(raw_res, load_res, atol=1e-6))