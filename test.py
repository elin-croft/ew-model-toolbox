import sys
import ew_model as models
import torch

torch.set_default_device("mps")
dummy = torch.randn((1,3,224,224))

model = models.BACKBONE.get('Vgg')()
out = model(dummy)
print(out.shape)
print(models.RECMODEL.get_cls_map())
print(models.LOSS.get_cls_map())
print(models.PARSERS.get_cls_map())

args = models.PARSERS.get("VggArgs")()
res = args.format_args()
print(res)