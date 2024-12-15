import torch
import torch.functional
import torch.nn as nn
from ew_model.builder import BACKBONE


class BaseCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, 
                 kernel_size=(3, 3), stride=1, padding=(1,1),
                 norm=None, activate=None, pooling=None, **kwargs) -> None:
        super(BaseCNN, self).__init__()
        cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding, **kwargs)
        self.module = nn.Sequential()
        modules = [cnn]
        # TODO: register norm act and pooling layers
        if norm == "BN":
            modules.append(nn.BatchNorm2d(out_channels))
        if activate == "Relu":
            modules.append(nn.ReLU(inplace=True))

        if pooling is not None:
            pool_args = pooling.split("_")
            pool_layer, pool_args = pool_args[0], tuple(map(int, pool_args[1].split("-")))
            if pool_layer == "max":
                modules.append(nn.MaxPool2d((pool_args[0],pool_args[1]), pool_args[2]))
        self.module = nn.Sequential(*modules)
    
    def forward(self, x):
        out = x
        return self.module(out)

@BACKBONE.register_module()
class Vgg(nn.Module):
    def __init__(self, 
        kernel_size = 3,
        channels = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
        stride=1,
        padding=1,
        norm="BN",
        activation="Relu",
        pooling="max_2-2-2",
        pooling_position=[1, 3, 7, 11, 15]
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        def convert2list(arg, name, arg_type, match_object=channels):
            if isinstance(arg, list):
                if len(arg) != len(match_object) and len(arg) > 1:
                    raise Exception(f"{name} is a list, but it's length does not match length of the channels")
                elif len(arg) == 1:
                    return [arg[0]]*len(match_object)
                else:
                    return arg

            elif isinstance(arg, arg_type):
                return [arg] * len(match_object)
        
        # prepare args
        kernels = convert2list(kernel_size, "kernels", int)
        strides = convert2list(stride, "stride", int)
        paddings = convert2list(padding, "padding", int)
        norms = convert2list(norm, "norm", str)
        activations = convert2list(activation, "activation", str)
        poolings = convert2list(pooling, "pooling", int, pooling_position)
        pooling_position=set(pooling_position)

        # make layers
        convs = self._make_layers(kernels=kernels, channels=channels, strides=strides, paddings=paddings, 
                                  norms=norms, activations=activations,
                                  poolings=poolings, pooling_position=pooling_position)
        # add layers
        for conv in convs:
            self.convs.append(conv)
    
    def _make_layers(self, kernels, channels, strides, paddings, norms, activations, poolings, pooling_position):

        args = [kernels, channels, strides, paddings, norms, activations] 
        convs = []
        in_channel = 3
        j = 0
        for i, (kernel, channel, stride, padding, norm, act) in enumerate(zip(*args)):
            out_channel = channel
            pool = None
            if i in pooling_position:
                pool = poolings[j]
                j += 1
            conv = BaseCNN(kernel_size=kernel, in_channels=in_channel, out_channels=out_channel, stride=stride,
                       norm=norm, activate=act, pooling=pool, padding=padding)
            convs.append(conv)
            in_channel = out_channel
        return convs

    def forward(self, x):
        out = x
        for layer in self.convs:
            out = layer(out)
        return out
