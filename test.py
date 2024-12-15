import sys
import ew_model as models
import torch

torch.set_default_device("mps")
dummy = torch.randn((10,3,224,224))

from configs.vgg_config import VGG
args = models.PARSERS.get("VggArgs")(VGG)
res = args.format_args()
vgg = models.BACKBONE.get('Vgg')(**res)
out = vgg(dummy)
print(out.shape)