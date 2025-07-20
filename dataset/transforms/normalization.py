import torch
import numpy as np
import torch.nn as nn

class Normalization(nn.Module):
    def __init__(self, min_val=0, max_val=255):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor) and not isinstance(x, np.ndarray):
            raise TypeError("Input must be a torch.Tensor or numpy.ndarray")
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return (x - self.min_val) / (self.max_val - self.min_val)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(min_val={self.min_val}, max_val={self.max_val})"