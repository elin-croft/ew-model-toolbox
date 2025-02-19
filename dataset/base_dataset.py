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
        self.__block_config = None
        self.__classes = None
        self.__classes_to_idx = None

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
            real_data, _ = data.feature
            real_data_device = self.__set_device(real_data, device)
            data.feature = real_data_device

        else:
            logging.warning(f"Unsupported data type: {type(data)}")

        return data

    def to(self, device, **kwargs):
        self.datas = self.__set_device(self.datas, device, **kwargs)
        self.targets = self.__set_device(self.targets, device, **kwargs)
        return self

    @property
    def classes(self):
        if hasattr(self, "__classes"):
            return self.__classes
        return None
    
    @classes.setter
    def classes(self, classes):
        self.__classes = classes

    @property
    def class_to_idx(self):
        if hasattr(self, "__class_to_idx"):
            return self.__class_to_idx
        return None
    
    @class_to_idx.setter
    def class_to_idx(self, class_to_idx):
        self.__class_to_idx = class_to_idx

    @property
    def block_config(self):
        if hasattr(self, "__block_config"):
            return self.__block_config
        return None
    
    @block_config.setter
    def block_config(self, block_config):
        self.__block_config = block_config