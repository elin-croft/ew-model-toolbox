import sys, os
import ew_model as models
import torch

#torch.set_default_device("mps")
device = torch.device("mps")
dummy = torch.randn((10,3,224,224)).to(device)
label = torch.randint(0,1000,(10,)).to(device)

import configs.vgg_config as vgg_config
args = vgg_config.compose()
model = models.build_model(args['model_cfg']).to(device)
out = model(dummy)
# loss = models.build_loss(args['loss_cfg']).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# l = loss(out, label)
# optimizer.zero_grad()
# l.backward()
# optimizer.step()
# print(l)
from dataset import CsvDataset, CsvRecDataset

def hook(obj:object):
    obj.__setattr__("label_map",{str(i): i for i in range(1000)})
    #print(obj.label_map)

from torch.utils.data import DataLoader
from dataset import DATASET
# dataset = CsvDataset(path="/Users/elinwang/Documents/dataset.csv", hook=hook).to(device)
dataset = CsvRecDataset(path="/Users/elinwang/Documents/rec_dataset.csv", hook=hook, is_map=True).to(device)

# can't set default device to gpu when shuffle is True, for the generator is not supported
#generator = torch.Generator(device=device)
data = DataLoader(dataset, batch_size=1, shuffle=True)
for img, label in data:
    # img = img.to("mps")
    # label = label.to("mps")
    print(img)
    print(label)
