from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class BaseDataset(Dataset):
    def __init__(self,
        path: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.path = path
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
