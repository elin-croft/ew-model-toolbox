import torch
class BlockConfig:
    def __init__(self,
        block_id=None, block_name=None,
        data_type = torch.float32,
        size: int = None, emb_size: int = None, layer_name: str = None, layer_args: dict = None,
        allow_shorter = True, default_value = 0
    ):
        self.block_id = block_id
        self.block_name = block_name
        self.data_type = data_type
        self.size = size
        self.emb_size = emb_size
        self.layer_name = layer_name
        self.layer_args = layer_args
        self.allow_shorter = allow_shorter
        self.default_value = default_value
        