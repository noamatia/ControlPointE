import os
import math
import torch
from typing import Optional
from safetensors.torch import save_file
from point_e.models.transformer import PointDiffusionTransformer


class LoRALinearModule(torch.nn.Module):
    def __init__(
        self,
        lora_name: str,
        org_module: torch.nn.Module,
        lora_dim: int = 4,
        alpha: int = 1,
    ):
        super().__init__()
        self.multiplier = 1
        self.org_module = org_module
        self.lora_name = lora_name
        self.scale = alpha / lora_dim
        self.lora_down = torch.nn.Linear(org_module.in_features, lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(lora_dim, org_module.out_features, bias=False)
        self.register_buffer("alpha", torch.tensor(alpha))
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRA(torch.nn.Module):
    def __init__(
        self,
        model: PointDiffusionTransformer,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.lora_scale = 1
        self.alpha = alpha
        self.lora_dim = rank
        self.lora_modules = self.create_modules(model)
        print(f"PointDiffusionTransformer LoRA: {len(self.lora_modules)} modules.")
        self.apply_lora_modules()
        del model
        torch.cuda.empty_cache()

    def create_modules(self, model: PointDiffusionTransformer) -> list:
        loras = []
        for name, module in model.named_modules():
            if model.is_trainable_param(name) and module.__class__.__name__ == "Linear":
                lora_name = name.replace(".", "_")
                lora = LoRALinearModule(lora_name, module, self.lora_dim, self.alpha)
                loras.append(lora)
        return loras

    def apply_lora_modules(self):
        lora_names = set()
        for lora in self.lora_modules:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}"
            lora_names.add(lora.lora_name)
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def prepare_optimizer_params(self):
        all_params = []
        if self.lora_modules:
            params = []
            [params.extend(lora.parameters()) for lora in self.lora_modules]
            param_data = {"params": params}
            all_params.append(param_data)
        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.lora_modules:
            lora.multiplier = self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.lora_modules:
            lora.multiplier = 1

    def print_parameters_status(self):
        for name, param in self.named_parameters():
            print(
                f"name: {name}, shape: {param.shape}, req grad: {param.requires_grad}"
            )