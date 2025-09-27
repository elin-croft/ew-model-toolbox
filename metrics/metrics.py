import torch
from .builder import METRICS

@METRICS.register_module()
def acc(pred:torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum().item()
    return correct / len(target)