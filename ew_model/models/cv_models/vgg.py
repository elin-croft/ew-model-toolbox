import torch
import torch.nn as nn
from ..builder import BACKBONE


@BACKBONE.register_module()
class Vgg(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)