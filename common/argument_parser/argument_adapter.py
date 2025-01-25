import argparse
from abc import ABCMeta, abstractmethod


class BaseParser(metaclass=ABCMeta):
    def __init__(self):
        self.arg_name = self.__class__.__name__
        self.module_name = None

    def to_pair_list(self):
        return [(k, v) for k, v in self.__dict__.items()]

    def to_map(self):
        return {k:v for k, v in self.__dict__.items()}

    def get_parser(self)->argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=f"{self.arg_name} configs")
        return parser

    @abstractmethod
    def add_argument(self, parser: argparse.ArgumentParser):
        pass

    def format_args(self, to_map=True):
        assert self.module_name is not None
        if to_map:
            return self.to_map()
        else:
            return self.to_pair_list()
    
    def __call__(self, args=None, to_map=True, **kwds):
        parser = self.get_parser()
        parser.add_argument("--module_name", type=str, default=None, help="model name")
        self.add_argument(parser)
        namespace = parser.parse_args(args=args)
        for k, v in vars(namespace).items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise Exception(f"variable {k} is not in class {self.arg_name}")
        if to_map:
            res = {}
            for k, v in vars(namespace).items():
                res[k] = v
            return res
        else:
            return [(k, v) for k, v in vars(namespace).items()]
