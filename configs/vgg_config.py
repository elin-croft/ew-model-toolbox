from ew_model.builder import PARSERS

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

def compose():
    backbone_args = PARSERS.build_args("VggBackboneArgs", VGGBACKBONE)
    head_args = PARSERS.build_args("VggHeadArgs", HEAD)
    train_cfg = dict(
        device = "mps"
    )
    model_args = dict(
        module_name="Vgg",
        backbone=backbone_args,
        head=head_args
    )
    loss_args = dict(
        module_name="CrossEntropy"
    )
    args = dict(
        model=model_args,
        loss=loss_args,
        train_cfg=train_cfg
    )
    return args