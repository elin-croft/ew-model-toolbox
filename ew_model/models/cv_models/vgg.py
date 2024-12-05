import torch
import torch.functional
import torch.nn as nn
from ...builder import BACKBONE


class BaseCNN(nn.Module):
    def __init__(self, norm=True, activate=True, pooling=True, 
                 in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=(1,1), **kwargs) -> None:
        super(BaseCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding, **kwargs)
        self.norm_layer = None
        self.activate_layer = None
        self.norm = norm
        self.activate =activate
        if norm:
            self.norm_layer = nn.BatchNorm2d(out_channels)
        if activate:
            self.activate_layer = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm_layer(out)
        if self.activate:
            out = self.activate_layer(out)
        return out

@BACKBONE.register_module()
class Vgg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = BaseCNN()
    
    def forward(self, x):
        out = self.layers(x)
        return out
