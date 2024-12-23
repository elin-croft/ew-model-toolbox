import sys
import ew_model as models
import torch

torch.set_default_device("mps")
dummy = torch.randn((10,3,224,224))
label = torch.randint(0,1000,(10,))

import configs.vgg_config as vgg_config
args = vgg_config.compose()
model = models.build_classifier(args['model'])
out = model(dummy)
loss = models.build_loss(args['loss'])
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
l = loss(out, label)
optimizer.zero_grad()
l.backward()
optimizer.step()
print(out.shape)