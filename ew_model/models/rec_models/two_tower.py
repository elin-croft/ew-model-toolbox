from typing import List, Dict

import torch
import torch.nn as nn
from ew_model.builder import MODEL, build_input
from common.feature import BlockConfig

@MODEL.register_module()
class TwoTowerModel(nn.Module):
    def __init__(self,
        input_cfg: Dict=None,
        block_config: Dict[int, BlockConfig]=None,
        user_fc: List[int] = [512, 256, 128],
        item_fc:List[int] = [512, 256, 128],
        **kwargs) -> None:
        super().__init__()
        self.user_fc = user_fc
        self.item_fc = item_fc
        self.user_tower = nn.Sequential()
        self.item_tower = nn.Sequential()
        self.user_block = {k: v for k, v in block_config.items() if v.common}
        self.item_block = {k: v for k, v in block_config.items() if not v.common}
        self.user_input = None
        self.item_input = None
        self.build(input_cfg)
    
    def make_layer(self, in_channel, fc_sizes):
        layers = []
        for size in fc_sizes:
            layers.append(nn.Linear(in_channel, size))
            # disable batchnorm when batch size is 1 and model is in train mode
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            in_channel = size
        return nn.Sequential(*layers)
    
    def build_input(self, input_cfg: Dict):
        user_input_cfg_base = input_cfg.get("user_input")
        item_input_cfg_base = input_cfg.get("item_input")
        user_input_cfg = dict(
            **user_input_cfg_base,
            block_config = self.user_block
        )
        item_input_cfg = dict(
            **item_input_cfg_base,
            block_config = self.item_block
        )
        self.user_input = build_input(user_input_cfg)
        self.item_input = build_input(item_input_cfg)

    def build(self, input_cfg: Dict):
        self.build_input(input_cfg)
        user_emb_dim = self.user_input.shape
        item_emb_dim = self.item_input.shape
        self.user_tower = self.make_layer(user_emb_dim, self.user_fc)
        self.item_tower = self.make_layer(item_emb_dim, self.item_fc)

    def forward(self, input_dict: Dict[int, torch.Tensor]):
        user_input = self.user_input(input_dict)
        item_input = self.item_input(input_dict)
        user_emb = self.user_tower(user_input)
        item_emb = self.item_tower(item_input)

        # dot product
        output = torch.mul(user_emb, item_emb).sum(dim=1)
        return output
