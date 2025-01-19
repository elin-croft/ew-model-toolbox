from typing import Optional, Callable

import torch
from torch.utils.data import Dataset

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

    def __set_device(self, data, device):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
            return data

        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = self.__set_device(data[i], device)
            return data

        elif isinstance(data, dict):
            for k, v in data.items():
                v = self.__set_device(v, device)
                data[k] = v
            return data
        else:
            raise ValueError("data type should be tensor or list of tensor")

    def to(self, device):
        self.datas = self.__set_device(self.datas, device)
        self.targets = self.__set_device(self.targets, device)
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