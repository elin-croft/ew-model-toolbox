import sys
import ew_model as models
import torch

torch.set_default_device("mps")
dummy = torch.randn((10,3,224,224))

import configs.vgg_config as vgg_config
args = vgg_config.compose()
model = models.CLASSIFIER.build(args)
out = model(dummy)
print(out.shape)