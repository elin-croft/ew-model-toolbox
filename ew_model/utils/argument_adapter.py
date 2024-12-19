import argparse
from abc import ABCMeta, abstractmethod


class Adapter(metaclass=ABCMeta):
    def __init__(self):
        self.arg_name = self.__class__.__name__
        self.module_name = None

    def to_pair_list(self):
        return [(k, v) for k, v in self.__dict__.items()]

    def to_map(self):
        return {k:v for k, v in self.__dict__.items()}

    def parse_args(self, args=None):
        parser = self.get_parser()
        args = parser.parse_args(args=args)
        self.module_name = args.module_name
        return args
    
    def get_parser(self)->argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=f"{self.arg_name} configs")
        parser.add_argument("--module_name", type=str, default=None, help="model name")
        return parser

    def format_args(self, to_map=True):
        assert self.module_name is not None
        if to_map:
            return self.to_map()
        else:
            return self.to_pair_list()
    
    def __call__(self, args=None, **kwds):
        arg = self.parse_args(args=args)
        self.module_name = arg.module_name
        return self.format_args()