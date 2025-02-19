import logging
from typing import List, Dict
from collections import OrderedDict

import torch
import torch.nn.functional as F

from .block_config import BlockConfig

def data_padding_1d(data: torch.Tensor, size, allow_shorter, padding_value):
    if data.ndim > 1:
        raise Exception("cannot padding a tensor that has two more dim")

    if allow_shorter and size > data.shape[-1]:
        padding_size = size - data.shape[-1]
        F.pad(data, (0, padding_size), mode='constant', value=padding_value)
        return data
    else:
        return data

class FeatureItem:
    def __init__(self,
        feature_vector: torch.Tensor = None,
        ordered_feature: dict = None,
        label: List[torch.Tensor] = None,
        is_map: bool = False
    ):
        self.feature_vector = feature_vector # raw feature
        self.ordered_feature = ordered_feature
        self.ori_feature_vector = None # raw feature
        self.ori_ordered_feature = None
        self.label = label
        self.is_map = is_map

    @property
    def feature(self):
        if self.is_map:
            return self.ordered_feature, self.label
        return self.feature_vector, self.label
    
    @feature.setter
    def feature(self, feature):
        if self.is_map:
            if not isinstance(feature, OrderedDict):
                msg = f"this feature item works as map, feature should be OrderedDict, but got {type(feature)}"
                raise ValueError(msg)
            self.ordered_feature = feature
        else:
            if not isinstance(feature, torch.Tensor):
                msg = f"this feature item works as vector, feature should be torch.Tensor, but got {type(feature)}"
                raise ValueError(msg)
            self.feature_vector = feature
    
    def fix_data(self, blk_cfgs: Dict[int, BlockConfig]):
        if not self.is_map:
            return
        new_data = OrderedDict()
        ori_data =  self.feature
        for k, v in ori_data.items():
            block_id = int(k)
            cfg = blk_cfgs.get(block_id, None)
            if cfg:
                size = cfg.size
                allow_shorter = cfg.allow_shorter
                default_value = cfg.default_value
                v = data_padding_1d(v, size, allow_shorter, default_value)
            else:
                logging.warning(f"block: {block_id} not in config")
            new_data[k] = v
        self.feature = new_data
        self.ori_ordered_feature = ori_data
