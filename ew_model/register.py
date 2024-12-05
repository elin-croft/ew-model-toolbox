import os, sys

class Register:
    def __init__(self, name):
        self.name = name
        self.__model_cls_map = dict()
    
    def get_module_by_name(self, name):
        try:
            module = self.__model_cls_map[name]
            return module
        except KeyError:
            raise KeyError(f"{name} hasn't been registered yet")

    def check_module(self, name):
        if self.__model_cls_map.get(name) is None:
            return False
        else:
            return True
    
    def get_cls_map(self):
        return self.__model_cls_map

    def __register(self, name, module):
        self.__model_cls_map[name] = module

    def register_module(self, module=None, name=None, force=False):
        if name is not None:
            # TODO: register module seperately
            pass

        if not force and self.check_module(name):
            raise RuntimeError(f"{name} has been registered")

        def __register(cls):
            name = cls.__name__
            self.__register(name, cls)
        return __register