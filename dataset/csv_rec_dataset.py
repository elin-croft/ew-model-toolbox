import os
import pickle
from typing import Optional, Callable
from collections import OrderedDict

import torch

from .csv_dataset import CsvDataset
from common.feature import FeatureItem
from .builder import DATASET

@DATASET.register_module()
class CsvRecDataset(CsvDataset):
    def __init__(self,
        path:str = None,
        is_relative:bool = True,
        transform:Optional[Callable] = None, target_transform:Optional[Callable] = None,
        hook:Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(path, is_relative, transform, target_transform, hook)
    
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
                        feature[str(block_id)] = data['feature'][str(block_id)]
                    label = torch.tensor(data.get("label", list(map(float, label.split(",")))),dtype=torch.float32)
                    featureItem = FeatureItem(ordered_feature=feature, label=label, is_map=True)
            self.datas.append(featureItem)
            self.targets.append(label)
    
    def fetch_data(self, index):
        featureItem, label = self.datas[index], self.targets[index]
        data, _ = featureItem.feature
        return data, label
