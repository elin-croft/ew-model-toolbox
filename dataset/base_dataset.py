from typing import Optional, Callable
import logging

import torch
from torch.utils.data import Dataset

from common.feature import FeatureItem

class BaseDataset(Dataset):
    def __init__(self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.datas = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        base get item mothod for dataset class
        """
        data, target = self.datas[index], self.targets[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target
    
    def __len__(self):
        return len(self.datas)

    def __set_device(self, data, device, **kwargs):
        if isinstance(data, torch.Tensor):
            data = data.to(device)

        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = self.__set_device(data[i], device, **kwargs)

        elif isinstance(data, dict):
            for k, v in data.items():
                v = self.__set_device(v, device, **kwargs)
                data[k] = v

        elif isinstance(data, FeatureItem):
            is_map = kwargs.get("is_map", )
            real_data, _ = data.get_feature(is_map=is_map)
            real_data_device = self.__set_device(real_data, device, is_map=is_map)
            data.set_feature(real_data_device, is_map=is_map)

        else:
            logging.warning(f"Unsupported data type: {type(data)}")

        return data

    def to(self, device, **kwargs):
        self.datas = self.__set_device(self.datas, device, **kwargs)
        self.targets = self.__set_device(self.targets, device, **kwargs)
        return self

    @property
    def classes(self):
        if hasattr(self, "classes"):
            return self.classes
        return None

    @property
    def class_to_idx(self):
        if hasattr(self, "class_to_idx"):
            return self.class_to_idx
        return None

    @property
    def block_config(self):
        if hasattr(self, "block_config"):
            return self.block_config
        return None