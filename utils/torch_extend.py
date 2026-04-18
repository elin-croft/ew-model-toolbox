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

def load_partilay_state_dict(old_param: torch.Tensor, new_param: torch.Tensor, initial_type='xavier'):
    """
    Loads a state dictionary into a module, allowing for partial matches.

    Args:
        old_param (torch.Tensor): The parameter tensor to be updated.
        new_param (torch.Tensor): The new parameter tensor.
        initial_type (str): The initialization type for the updated parameter tensor.
    Returns:
        torch.Tensor: The updated parameter tensor.
    """
    _, in_feature_new = new_param.shape
    in_feature_old = old_param.shape[1]
    with torch.no_grad():
        if in_feature_new > in_feature_old:
            if initial_type == 'xavier':
                new_param[:, :in_feature_old] = old_param
                nn.init.xavier_uniform_(new_param[:, in_feature_old:])
            else:
                new_param[:, :in_feature_old] = old_param
                nn.init.kaiming_uniform_(new_param[:, in_feature_old:], a=np.sqrt(5))
        else:
            new_param = old_param[:, :in_feature_new]
    return new_param

