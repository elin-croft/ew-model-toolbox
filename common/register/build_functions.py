import inspect

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .register import Register

def build_from_cfg(cfg: dict, register: 'Register'):
    """
    Build a module from the configuration dictionary.
    The cfg should contain 'module_name' and other necessary parameters.
    """
    # TODO: cfg should be a dict that each key is a model or layer or loss name

    if "module_name" not in cfg.keys():
        raise KeyError("please set module_name before you build it, type should be map<str, str>")
    module_name = cfg.get('module_name') 
    if module_name is None:
        raise KeyError(f"{module_name} is not in register, please check if your model has been propertly registered")
    args = cfg.copy()
    args.pop('module_name')

    try:
        obj_cls = register.get(module_name)
    except KeyError as e:
        print(f"build {register.name} error: {str(e)}")
    if inspect.isclass(obj_cls):
        return obj_cls(**args)
    elif inspect.isfunction(obj_cls):
        return obj_cls
    else:
        raise TypeError(f"{module_name} is not a class or function, please check if your model has been propertly registered")