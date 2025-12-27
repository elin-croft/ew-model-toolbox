import os
from .model_param import model_cfg
from .block_config import config

def hook(obj:object):
    obj.block_config={i.block_id: i for i in config}

loss_cfg=dict(
    module_name="BinaryCrossEntropy"
)
dataset_cfg=dict(
    module_name="CsvRecDataset",
    path=os.path.join(os.path.expanduser("~"),"Documents/rec_dataset.csv"),
    is_map=True,
    hook=hook
)
dataloader_cfg=dict(
    module_name="DataLoader",
    batch_size=1,
    shuffle=True,
)
data_setter_cfg=dict(
    module_name="default_data_device_setter",
)
train_cfg = dict(
    device = "mps",
    batch_size=512,
    optimizer=dict(
        module_name="Adam",
        lr=0.01
    )
)
test_cfg = dict(
    device = "mps"
)
def Compose():
    args = dict(
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        data_cfg=dict(
            dataset_cfg=dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            data_setter_cfg=data_setter_cfg
        ),
        # train param config
        train_cfg=train_cfg,
        # test param config
        test_cfg=test_cfg
    )
    return args