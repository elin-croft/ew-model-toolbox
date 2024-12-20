import argparse
from .parsers import kernel_size_parser, tuple_list_parser
from .argument_adapter import Adapter
from ew_model.builder import PARSERS
from typing import List, Tuple

@PARSERS.register_module()
class VggBackboneArgs(Adapter):
    def __init__(self):
        super().__init__()
        self.kernel_size = None
        self.channels = None
        self.padding = None
        self.norm = None
        self.activation = None
        self.pooling = None
        self.stride = None
        self.pooling_position = None

    def add_argument(self, parser):
        parser.add_argument("--kernel_size", "-k", type=kernel_size_parser, default=3, 
                            help="kernel size for basic cnn modules")
        parser.add_argument("--channels", type=int, nargs="*", default=[64],
                            help="in and out channels, [(in, out)...]")
        parser.add_argument("--stride", type=int, nargs="*", default=[1])
        parser.add_argument("--padding", type=kernel_size_parser, default=1, help="paddings")
        parser.add_argument("--norm", type=str, default=["BN"], nargs="*", help="normlization layers")
        parser.add_argument("--activation", type=str, default="Relu", nargs="*", help="activate layers")
        parser.add_argument("--pooling", type=str, default="max_2-2-2", nargs="*", 
                            help="pooling layers, type_h_w_stride")
        parser.add_argument("--pooling_position", type=int, nargs="*", default=[1, 3, 7, 11, 15], help="pooling position")


@PARSERS.register_module()
class VggHeadArgs(Adapter):
    def __init__(self):
        super().__init__()
        self.in_channel=None
        self.hidden_channels=None
        self.class_num=None
    
    def add_argument(self, parser):
        parser.add_argument("--in_channel", type=int,  default=3, help="in channel")
        parser.add_argument("--hidden_channels", type=int, default=[4096], nargs="+", help="hidden layer channels")
        parser.add_argument("--class_num", type=int, default=1000, help="class num")
