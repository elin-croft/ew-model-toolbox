from .model_param import model_cfg
from .block_config import config

def hook(obj:object):
    obj.block_config={i.block_id: i for i in config}

def Compose():
    import os
    args = dict(
        model_cfg=model_cfg,
        loss_cfg=dict(
            module_name="CrossEntropy"
        ),
        dataset_cfg=dict(
            module_name="CsvRecDataset",
            path=os.path.join(os.path.expanduser("~"),"Documents/rec_dataset.csv"),
            is_map=True,
            hook=hook
        ),
        # train param config
        train_cfg = dict(
            device = "mps",
            batch_size=512,
            optimizer=dict(
                module_name="Adam",
                lr=0.01
            )
        ),
        # test param config
        test_cfg = dict(
            device = "mps"
        )
    )
    return args