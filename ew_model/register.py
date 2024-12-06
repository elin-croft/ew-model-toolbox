import os, sys

class Register:
    def __init__(self, name):
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
    
    def build(self, cfg: dict):
        # TODO: cfg should be a dict that each key is a model or layer or loss name
        model_name = cfg.get('model_name') 
        args = cfg.copy()
        args.pop('model_name')
        try:
            obj_cls = self.get(model_name)
        except Exception as e:
            print(f"build {self.name} error: {str(e)}")
        return obj_cls(**args)