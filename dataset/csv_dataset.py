import os
from typing import Optional, Callable

import csv
import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from .builder import DATASET

@DATASET.register_module()
class CsvDataset(BaseDataset):
    def __init__(self,
        path: str = None, is_relative:bool = True,
        transform:Optional[Callable] = None, target_transform:Optional[Callable] = None,
        hook:Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(transform, target_transform)
        self.is_relative = is_relative
        self.get_data(path)
        if hook is not None:
            hook(self)
    
    def load_csv(self, path):
        paths = []
        labels = []
        prefix = ""
        if self.is_relative and path.endswith(".csv"):
            prefix = os.path.split(path)[0]

        with open(path, 'r', encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(prefix, row["path"])
                label = row["label"]
                paths.append(path)
                labels.append(label)

        return paths, labels
    
    def get_data(self, path):
        # reset datas and targets
        self.datas = []
        self.targets = []
        data_paths, data_labels = self.load_csv(path)
        for path, label in zip(data_paths, data_labels):
            img = cv2.imread(path)
            # np(h,w,c) -> tensor(1, c,h,w)
            img = torch.tensor(np.transpose(img, (2, 0, 1)))
            label = torch.tensor(list(map(int, label.split(","))))
            self.datas.append(img)
            self.targets.append(label)

        #self.datas = torch.concat(self.datas, dim=0)
