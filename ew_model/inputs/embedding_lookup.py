import torch
import torch.nn as nn

from ew_model.builder import INPUT

"""
one hot feature to embedding
"""

@INPUT.register_module()
class EmbeddingLookup(nn.Module):
    def __init__(self, size: int = None, emb_size=8):
        super().__init__()
        self.size = size
        self.emb_size = emb_size
        self.layer = nn.Embedding(size, emb_size)
    
    @property
    def shape(self):
        return self.emb_size

    def forward(self, x):
        index = torch.argmax(x, dim=1, keepdim=False)
        return self.layer(index)
