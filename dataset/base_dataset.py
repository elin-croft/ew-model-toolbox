from typing import Optional, Callable
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
