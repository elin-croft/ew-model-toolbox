# from .models import BACKBONE, RECMODEL
# from .losses import LOSS
from .register import Register
import torch.nn as nn

RECMODEL=Register("rec_models")
BACKBONE=Register("backbone")
LOSS=Register("loss")
PARSERS=Register("parsers")
NORMLIZATION=Register("normlization")
NORMLIZATION.register_module("BN", nn.BatchNorm2d, force=True)

