import argparse
from .parsers import kernel_size_parser
from .argument_adapter import Adapter
from ew_model.builder import PARSERS

@PARSERS.register_module()
class VggArgs(Adapter):
    def __init__(self, args=None):
        self.kernel_size = None
        self.in_channels = None
        self.out_channels = None
        self.padding = None
        self.parse_args(args)

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser(description="cnn moudle configs")
        parser.add_argument("--kernel_size", "-k", type=kernel_size_parser, default=3, 
                            help="kernel size for basic cnn modules")
        parser.add_argument("--in_channels", default=[3], nargs="+", help="in channel")
        parser.add_argument("--out_channels", default=[3], nargs="+", help="out channel")
        parser.add_argument("--paddings", type=kernel_size_parser, default=1,help="paddings")
        args = parser.parse_args(args=args)
        self.kernel_size = args.kernel_size
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.padding = args.paddings

    def format_args(self, to_map=True):
        if to_map:
            return self.to_map()
        else:
            return self.to_pair_list()
