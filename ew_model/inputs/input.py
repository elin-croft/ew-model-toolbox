from typing import Dict

import torch
import torch.nn as nn

from ew_model.builder import INPUT, build_input
from common.feature import BlockConfig

@INPUT.register_module()
class BaseInput(nn.Module):
    def __init__(self, block_config: Dict[int, BlockConfig] = None):
        super().__init__()
        self.block_ids = sorted([int(i) for i in block_config.keys()])
        self.block_config = block_config
        self.layer_map = nn.ModuleDict()
    
    def build(self):
        for bid, config in self.block_config.items():
            extra_arg = config.layer_args if config.layer_args is not None else {}
            args = dict(
                module_name = config.layer_name,
                size = config.size,
                emb_size = config.emb_size,
                **extra_arg
            )
            self.layer_map[bid] = build_input(args)
    
    def forward(self, x: dict):
        out = []
        for block_id in self.block_ids:
            value = x.get(int(block_id))
            layer = self.layer_map[block_id]
            out.append(layer(value))
        out = torch.cat(out, dim=1)

        return out