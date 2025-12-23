import os, sys
from abc import ABCMeta
import inspect
from typing_extensions import deprecated
import logging
from .build_functions import *

class Register:
    def __init__(self, name, build_func=None, parent=None):
        self.name = name
        self.__model_cls_map = dict()
        if build_func is None:
            self.build_func = build_from_cfg
        else:
            self.build_func = build_func
    
    def get(self, name):
        module = self.__model_cls_map[name]
        if module is None:
            raise KeyError(f"{name} hasn't been registered yet")
        return module

    def has_module(self, name):
        if self.__model_cls_map.get(name) is None:
            return False
        else:
            return True
    
    def get_cls_map(self):
        return self.__model_cls_map

    def __register(self, name, module):
        self.__model_cls_map[name] = module
        return module

    def register_module(self, name=None, module=None, force=False):
        if name is not None and module is not None:
            has = self.has_module(name=name)
            if has:
                if force:
                    msg = f"module {name} will be registered in force and the old module will be replaced"
                    logging.warning(msg)
                    return self.__register(name, module)
                else:
                    raise KeyError(f"{name} has been registered")
            else:
                return self.__register(name, module)

        if not force and self.has_module(name):
            raise KeyError(f"{name} has been registered")

        def __register(cls):
            name = cls.__name__
            self.__register(name, cls)
            return cls
        return __register
    
    def build(self, cfg: dict):
        # TODO: cfg should be a dict that each key is a model or layer or loss name

        # if "module_name" not in cfg.keys():
        #     raise KeyError("please set module_name before you build it, type should be map<str, str>")
        # module_name = cfg.get('module_name') 
        # if module_name is None:
        #     raise KeyError(f"{module_name} is not in register, please check if your model has been propertly registered")
        # args = cfg.copy()
        # args.pop('module_name')

        # try:
        #     obj_cls = self.get(module_name)
        # except KeyError as e:
        #     print(f"build {self.name} error: {str(e)}")
        # if inspect.isclass(obj_cls):
        #     return obj_cls(**args)
        # elif inspect.isfunction(obj_cls):
        #     return obj_cls
        # else:
        #     raise TypeError(f"{module_name} is not a class or function, please check if your model has been propertly registered")
        return self.build_func(cfg, self)

    @deprecated("build_args is deprecated, please use dict as config instead of --args string")
    def build_args(self, name, args: str):
        arg_cls = self.get(name=name)
        if arg_cls is None:
            raise KeyError(f"{name} is not in register, please check if your model has been propertly registered")
        
        args = args.replace("\n", " ").strip().split(" ")
        obj = arg_cls()
        argMap = obj(args)
        if "arg_name" in argMap.keys():
            argMap.pop('arg_name')
        return argMap


PARENT_REGISTER=Register("parent_register")