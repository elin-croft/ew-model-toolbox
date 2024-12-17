import argparse
from .parsers import kernel_size_parser, tuple_list_parser
from .argument_adapter import Adapter
from ew_model.builder import PARSERS
from typing import List, Tuple

@PARSERS.register_module()
class VggBackboneArgs(Adapter):
    def __init__(self, args=None):
        super().__init__()
        self.model_name = None
        self.kernel_size = None
        self.channels = None
        self.padding = None
        self.norm = None
        self.activation = None
        self.pooling = None
        self.parse_args(args)

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser(description="cnn moudle configs")
        parser.add_argument("--model_name", type=str, default=None, help="model name")
        parser.add_argument("--kernel_size", "-k", type=kernel_size_parser, default=3, 
                            help="kernel size for basic cnn modules")
        parser.add_argument("--channels", type=int, nargs="*", default=[64],
                            help="in and out channels, [(in, out)...]")
        parser.add_argument("--stride", type=int, nargs="*", default=[1])
        parser.add_argument("--paddings", type=kernel_size_parser, default=1, help="paddings")
        parser.add_argument("--norms", type=str, default=["BN"], nargs="*", help="normlization layers")
        parser.add_argument("--activations", type=str, default="Relu", nargs="*", help="activate layers")
        parser.add_argument("--poolings", type=str, default="max_2-2-2", nargs="*", help="pooling layers, type_h_w_stride")
        args = parser.parse_args(args=args)

        self.model_name = args.model_name
        self.kernel_size = args.kernel_size
        self.channels = args.channels
        self.padding = args.paddings
        self.norm = args.norms
        self.activation = args.activations
        self.pooling = args.poolings

@PARSERS.register_module()
class VggHeadArgs(Adapter):
    def __init__(self, args=None):
        super().__init__()
        self.model_name = None
        self.in_channel=None
        self.hidden_channels=None
        self.class_num=None
        self.parse_args(args=args)
    
    def parse_args(self, args=None):
        parser = argparse.ArgumentParser(description="vgg classifation head args")
        parser.add_argument("--model_name", type=str, default=None, help="model name")
        parser.add_argument("--in_channel", type=int,  default=3, help="in channel")
        parser.add_argument("--hidden_channels", type=int, default=[4096], nargs="+", help="hidden layer channels")
        parser.add_argument("--class_num", type=int, default=1000, help="class num")
        args = parser.parse_args(args=args)

        self.model_name = args.model_name
        self.in_channel = args.in_channel
        self.hidden_channels = args.hidden_channels
        self.class_num = args.class_num
        