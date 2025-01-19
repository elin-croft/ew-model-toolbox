import torch
import torch.nn as nn
from ew_model.builder import RECMODEL
from common.feature import BlockConfig

@RECMODEL.register_module()
class TwoTowerModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.user_tower = nn.ModuleList()
        self.item_tower = nn.ModuleList()
        self.user_input = dict()
        self.item_input = dict()
    
    def forward(self):
        pass
