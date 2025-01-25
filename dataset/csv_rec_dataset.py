import os
import pickle
from typing import Optional, Callable
from collections import OrderedDict

import torch

from .csv_dataset import CsvDataset
from common.feature import FeatureItem


class CsvRecDataset(CsvDataset):
    def __init__(self,
        path:str = None,
        is_relative:bool = True,
        is_map:bool = False,
        transform:Optional[Callable] = None, target_transform:Optional[Callable] = None,
        hook:Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(path, is_relative, transform, target_transform, hook)
        self.is_map = is_map
    
    def load_csv(self, path):
        return super().load_csv(path)

    def get_data(self, path):
        # reset datas and targets
        self.datas = []
        data_paths, data_labels = self.load_csv(path)
        for path, label in zip(data_paths, data_labels):
            # np(n,) -> tensor(1, n)
            featureItem = None
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, FeatureItem):
                    featureItem = data
                elif isinstance(data, dict):
                    block_ids = data["block_ids"]
                    block_ids = sorted(list(map(int, block_ids)))
                    feature = OrderedDict()
                    for block_id in block_ids:
                        feature[block_id] = data['feature'][str(block_id)]
                    label = torch.tensor(data.get("label", list(map(int, label.split(",")))))
                    featureItem = FeatureItem(ordered_feature=feature, label=label)
            self.datas.append(featureItem)
            self.targets.append(label)
    
    def __getitem__(self, index):
        featureItem, label = self.datas[index], self.targets[index]
        data, _ = featureItem.get_feature(is_map=self.is_map)
        return data, label
    
    def to(self, device):
        return super().to(device, is_map=self.is_map)
