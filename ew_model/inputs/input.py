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
        self.build()
    
    def build(self):
        for bid, config in self.block_config.items():
            extra_arg = config.layer_args if config.layer_args is not None else {}
            args = dict(
                module_name = config.layer_name,
                size = config.size,
                emb_size = config.emb_size,
                **extra_arg
            )
            self.layer_map[str(bid)] = build_input(args)

    @property
    def shape(self):
        return sum([i.shape for i in self.layer_map.values()])
    
    def forward(self, x: Dict[str, torch.Tensor]):
        out = []
        # TODO: handle miss feature
        try:
            for block_id in self.block_ids:
                bid = str(block_id)
                value = x[bid]
                layer = self.layer_map[bid]
                if value.ndim == 1:
                    value = value.unsqueeze(0)
                out.append(layer(value))
            out = torch.cat(out, dim=1)
        except KeyError as e:
            raise KeyError(f"block_id {block_id} not in input data")

        return out