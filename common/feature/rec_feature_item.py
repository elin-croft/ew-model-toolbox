from typing import List
from collections import OrderedDict

import torch

class FeatureItem:
    def __init__(self,
        feature: torch.Tensor = None,
        ordered_feature: OrderedDict = None,
        label: List[torch.Tensor] = None
    ):
        self.feature = feature # raw feature
        self.ordered_feature = ordered_feature
        self.label = label

    def get_feature(self, is_map=False):
        if is_map:
            return self.ordered_feature, self.label
        return self.feature, self.label
    
    def set_feature(self, feature, is_map=False):
        if is_map:
            self.ordered_feature = feature
        else:
            self.feature = feature