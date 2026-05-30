import math
import torch
import torch.nn as nn
from gpt_model import GPT


class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        out_dim, in_dim = linear.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model, r=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    for i in range(len(model.h)):
        block = model.h[i]
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            linear = getattr(block.attn, name)
            setattr(block.attn, name, LoRALinear(linear, r=r, alpha=alpha))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def save_lora_weights(model, path):
    state = {n: p for n, p in model.named_parameters() if "lora_" in n}
    torch.save(state, path)
    print(f"LoRA weights saved to {path}")


def load_lora_weights(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    print(f"LoRA weights loaded from {path}")
    return model
