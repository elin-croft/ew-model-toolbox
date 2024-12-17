from ew_model.builder import PARSERS

VGGBACKBONE = f"""
--model_name VggBackbone
--kernel_size 3
--channels 64 64 128 128 256 256 256 256 512 512 512 512 512 512 512 512
--stride 1
--padding 1
--norm BN
--activations Relu
--pooling max_2-2-2
"""

HEAD = f"""
--model_name Linear
--in_channel {eval("512*7*7")}
--hidden_channels 4096 4096 4096
--class_num 1000
"""


def compose():
    backbone_args = PARSERS.build_args("VggBackboneArgs", VGGBACKBONE)
    head_args = PARSERS.build_args("VggHeadArgs", HEAD)
    args = dict(
        model_name="Vgg",
        backbone=backbone_args,
        head=head_args
    )
    return args