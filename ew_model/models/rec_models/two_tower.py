import torch
import torch.nn as nn
from ..builder import RECMODEL

@RECMODEL.register_module()
class TwoTowerModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.user_tower = nn.ModuleList()
        self.user_tower.add_module
        self.item_tower = None
    
    def forward(self):
        pass
