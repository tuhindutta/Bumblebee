import torch.nn as nn
import copy
from ..gpt.gpt import GPTLanguageModel
from .linear_with_lora import LinearWithLoRA



def print_trainable_parameters(model: GPTLanguageModel) -> None:
    trainable_parameters = 0
    all_parameters = 0
    for _, param in model.named_parameters():
        all_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()

    print(
        f"All parameters: {all_parameters/1e6:.2f}M | "
        f"Trainable parameters: {trainable_parameters/1e6:.2f}M | "
        f"Trainable %: {100 * trainable_parameters / all_parameters:.2f}%"
    )


def get_lora_model(model: GPTLanguageModel, lora_config: dict, device: str) -> GPTLanguageModel:
    lora_model = copy.deepcopy(model)
    _replace_linear_layers_with_lora_layers(lora_model, lora_config)
    _freeze_non_lora_layers(lora_model)
    lora_model = lora_model.to(device)
    return lora_model


def _replace_linear_layers_with_lora_layers(module: nn.Module, lora_config: dict) -> None:
    rank = lora_config.get('rank', 4)
    alpha = lora_config.get('alpha', 8)

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(
                child, rank=rank, alpha=alpha))
        else:
            _replace_linear_layers_with_lora_layers(
                child, lora_config)


def _freeze_non_lora_layers(model: GPTLanguageModel) -> None:
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False