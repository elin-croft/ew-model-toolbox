import torch
import logging
from dataset.builder import DATA_HELPER

@DATA_HELPER.register_module()
def default_data_device_setter(data, device, **kwargs):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = default_data_device_setter(data[i], device)
    elif isinstance(data, dict):
        exclude_keys = kwargs.get("exclude_keys", [])
        for k, v in data.items():
            if k in exclude_keys:
                continue
            v = default_data_device_setter(v, device)
            data[k] = v
    else:
        logging.warning(f"Unsupported data type: {type(data)}")
    return data