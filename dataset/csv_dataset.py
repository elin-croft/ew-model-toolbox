import os
import csv
import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset

class CsvDataset(BaseDataset):
    def __init__(self,
        path = None, is_relative = False,
        transform = None, target_transform = None,
        **kwargs):
        super().__init__(path, transform, target_transform)
        self.is_relative = is_relative
    
    def load_csv(self):
        paths = []
        if self.is_relative:
            prefix = self.path
        else:
            prefix = ""

        with open(self.path, 'r', encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(prefix, row["path"])
                label = row["label"]
                paths.append(path)
                self.targets.append(label)
        return paths
    
    def get_data(self):
        img_paths = self.load_csv()
        for path in img_paths:
            img = cv2.imread(path)
            # np(h,w,c) -> tensor(1, c,h,w)
            img = torch.tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0)
            self.datas.append(img)
        self.datas = torch.concat(self.datas, dim=0)