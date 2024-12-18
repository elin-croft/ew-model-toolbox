import torch
import torch.nn as nn
from ew_model.builder import BACKBONE, CLS_HEAD, CLASSIFIER

@CLASSIFIER.register_module()
class Vgg(nn.Module):
    def __init__(self,
                 backbone,
                 head):
        super().__init__()
        self.backbone = BACKBONE.build(backbone)
        self.flatten = nn.Flatten()
        self.head = CLS_HEAD.build(head)
    
    def forward(self, x):
        out = x
        out = self.backbone(out)
        out = self.flatten(out)
        out = self.head(out)
        return out
    
    @torch.no_grad()
    def predict(self, x):
        return self.forward(x)