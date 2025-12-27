import torchvision
from ew_model.builder import PARSERS
from dataset import Compose as piplineCompose
import dataset.transforms as transforms

VGGBACKBONE = f"""
--module_name VggBackbone
--kernel_size 3
--channels 64 64 128 128 256 256 256 256 512 512 512 512 512 512 512 512
--stride 1
--padding 1
--norm BN
--activation Relu
--pooling max_2-2-2
--pooling_position 1 3 7 11 15
"""

HEAD = f"""
--module_name Linear
--in_channel {eval("512*7*7")}
--hidden_channels 4096 4096 4096
--class_num 1000
"""

def hook(obj:object):
    obj.__setattr__("label_map",{str(i): i for i in range(1000)})
    # print(obj.label_map)

#  python train.py --model_config_path configs/vgg_config.py
def Compose():
    import os
    backbone_args = PARSERS.build_args("VggBackboneArgs", VGGBACKBONE)
    head_args = PARSERS.build_args("VggHeadArgs", HEAD)
    model_args = dict(
        module_name="Vgg",
        backbone=backbone_args,
        head=head_args
    )
    loss_args = dict(
        module_name="CrossEntropy"
    )
    data_cfg=dict(
        dataset_cfg = dict(
            module_name="CsvDataset",
            path=os.path.join(os.path.expanduser("~"), "Documents/dataset.csv"),
            hook=hook,
            transform=piplineCompose([
                transforms.Normalization(min_val=0.0, max_val=255.0),
                transforms.Resize(size=(224, 224))
            ])
        ),
        dataloader_cfg = dict(
            module_name="DataLoader",
            batch_size=1,
            shuffle=True,
        ),
        data_setter_cfg=dict(
            module_name="default_data_device_setter",
        )
    )
    # train param config
    train_cfg = dict(
        device = "mps",
        batch_size=32,
        optimizer=dict(
            module_name="SGD",
            lr=0.01
        )
    )
    # test param config
    test_cfg = dict(
        device = "mps"
    )
    args = dict(
        model_cfg=model_args,
        loss_cfg=loss_args,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        test_cfg=test_cfg
    )
    return args