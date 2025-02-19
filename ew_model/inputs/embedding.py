import torch
import torch.nn as nn

from ew_model.builder import INPUT

@INPUT.register_module()
class Embedding(nn.Module):
    def __init__(self, size: int = None, emb_size=8):
        super().__init__()
        self.size = size
        self.emb_size = emb_size
        self.layer = nn.Linear(size, emb_size)
    
    @property
    def shape(self):
        return self.emb_size

    def forward(self, x):
        return self.layer(x)