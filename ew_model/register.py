import os, sys
from abc import ABCMeta

class Register:
    def __init__(self, name, parent=None):
        self.name = name
        self.__model_cls_map = dict()
    
    def get(self, name):
        module = self.__model_cls_map[name]
        if module is None:
            raise KeyError(f"{name} hasn't been registered yet")
        return module

    def check_module(self, name):
        if self.__model_cls_map.get(name) is None:
            return False
        else:
            return True
    
    def get_cls_map(self):
        return self.__model_cls_map

    def __register(self, name, module):
        self.__model_cls_map[name] = module

    def register_module(self, name=None, module=None, force=False):
        if name is not None and module is not None:
            if force or self.check_module(name=name):
                self.__model_cls_map[name]=module
            else:
                raise KeyError(f"{name} has been registered")

        if not force and self.check_module(name):
            raise KeyError(f"{name} has been registered")

        def __register(cls):
            name = cls.__name__
            self.__register(name, cls)
        return __register
    
    def build(self, cfg: dict):
        # TODO: cfg should be a dict that each key is a model or layer or loss name

        if "model_name" not in cfg.keys():
            raise KeyError("please set model_name before you build it, type should be map<str, str>")
        model_name = cfg.get('model_name') 
        if model_name is None:
            raise KeyError(f"{model_name} is not in register, please check if your model has been propertly registered")
        args = cfg.copy()
        args.pop('model_name')

        try:
            obj_cls = self.get(model_name)
        except KeyError as e:
            print(f"build {self.name} error: {str(e)}")
        return obj_cls(**args)
    def build_args(self, name, args: str):
        arg_cls = self.get(name=name)
        if arg_cls is None:
            raise KeyError(f"{name} is not in register, please check if your model has been propertly registered")
        
        args = args.replace("\n", " ").strip().split(" ")
        obj = arg_cls(args)
        argMap = obj.format_args()
        if "arg_name" in argMap.keys():
            argMap.pop('arg_name')
        return argMap


PARENT_REGISTER=Register("parent_register")