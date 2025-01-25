import logging
from typing import List
from collections import OrderedDict

import torch

class FeatureItem:
    def __init__(self,
        feature_vector: torch.Tensor = None,
        ordered_feature: dict = None,
        label: List[torch.Tensor] = None,
        is_map: bool = False
    ):
        self.feature_vector = feature_vector # raw feature
        self.ordered_feature = ordered_feature
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