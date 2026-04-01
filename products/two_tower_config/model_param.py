from .block_config import config

model_cfg=dict(
    module_name="TwoTowerModel",
    input_cfg = dict(
        user_input = dict(
            module_name = "BaseInput"
        ),
        item_input = dict(
            module_name = "BaseInput"
        ),
    ),
    block_config = {i.block_id: i for i in config},
    user_fc = [512, 256, 128],
    item_fc = [512, 256, 128]
)