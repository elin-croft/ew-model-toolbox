import torch
import torch.nn as nn
import torch.nn.functional as F
from ew_model.builder import LOSS

def cross_entropy(input, label, weight, reduce, reduction='mean'):
    """
    input: input logits
    label: sample label
    weight: sum weight for each class 
    """
    return F.cross_entropy(input=input, target=label, weight=weight, reduce=reduce, reduction=reduction)


@LOSS.register_module()
class CrossEntropy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def forward(self, x, label, weight=None, reduce=True, reduction='mean'):
        out = cross_entropy(input=x, label=label, weight=weight, reduce=reduce, reduction=reduction)
        return out