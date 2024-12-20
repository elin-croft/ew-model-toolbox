import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)