import torch
import torch.nn as nn
import numpy as np

def sequence_mask(lengths, max_len=None, dtype=torch.bool):
    """
    Creates a boolean mask for sequences based on their lengths.

    Args:
        lengths (torch.Tensor): A 1D tensor of shape (batch_size,)
                                containing the lengths of each sequence.
        max_len (int, optional): The maximum length of the sequences.
                                 If None, it will be inferred from the
                                 maximum value in 'lengths'. Defaults to None.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, max_len).
    """
    if max_len is None:
        max_len = lengths.max().item()

    # Create a tensor representing indices from 0 to max_len-1
    indices = torch.arange(max_len, device=lengths.device).unsqueeze(0)

    # Expand lengths to match the dimensions of indices for broadcasting
    lengths_expanded = lengths.unsqueeze(1)

    # Compare indices with lengths to create the mask
    mask = indices < lengths_expanded
    mask = mask.to(dtype)
    return mask

def as_string(data):
    pass

def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d):
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    w = linear.weight.data
    b = linear.bias.data if linear.bias is not None else torch.zeros_like(mean)

    scale = gamma / torch.sqrt(var + eps)
    w_fused = w * scale.unsqueeze(1)
    b_fused = (b - mean) * scale + beta

    fused_linear = nn.Linear(linear.in_features, linear.out_features)
    with torch.no_grad():
        fused_linear.weight.copy_(w_fused)
        fused_linear.bias.copy_(b_fused)
    return fused_linear

if __name__ == "__main__":
    x = torch.randn(2, 10, 4)
    linear = nn.Linear(4, 8).eval()
    bn = nn.BatchNorm1d(8).eval()
    y1 = bn(torch.permute(linear(x), (0, 2, 1))).permute(0, 2, 1)
    fused_linear = fuse_linear_bn(linear, bn)
    y2 = fused_linear(x)
    print(torch.max(torch.abs(y1 - y2)).item())
