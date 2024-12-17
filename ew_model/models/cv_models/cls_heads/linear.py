import torch
import torch.nn as nn
from ew_model.builder import CLS_HEAD

@CLS_HEAD.register_module()
class Linear(nn.Module):
    def __init__(self, in_channel=512*7*7,
                 hidden_channels=[4090, 4090, 4090],
                 class_num=1000
                 ):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channles = hidden_channels
        self.class_num = class_num
        layers = self._make_layers(in_channel, hidden_channels, class_num)
        self.linear_layers = nn.Sequential(*layers)

    def _make_layers(self, in_channel, hidden_channels, class_num):
        layers = []
        in_features = in_channel
        for out_features in hidden_channels:
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            in_features = out_features
        layers.append(nn.Linear(in_features=in_features, out_features=class_num))
        return layers

    def forward(self, x):
        out = x
        out = self.linear_layers(out)
        return out
