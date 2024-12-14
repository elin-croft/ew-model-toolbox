from abc import ABCMeta, abstractmethod


class Adapter(metaclass=ABCMeta):
    def __init__(self):
        pass
    def to_pair_list(self):
        return [(k, v) for k, v in self.__dict__.items()]
    def to_map(self):
        return {k:v for k, v in self.__dict__.items()}

    @abstractmethod
    def parse_args(self, args=None):
        raise Exception("this an abstract function, please override it")
    
    def format_args(self, to_map=True):
        if to_map:
            return self.to_map()
        else:
            return self.to_pair_list()