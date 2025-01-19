import torch
import torch.nn as nn
from ew_model.builder import MODEL, build_backbone, build_head

@MODEL.register_module()
class Vgg(nn.Module):
    def __init__(self,
                 backbone,
                 head):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.flatten = nn.Flatten()
        self.head = build_head(head)
    
    def forward(self, x):
        out = x
        out = self.backbone(out)
        out = self.flatten(out)
        out = self.head(out)
        return out
    
    @torch.no_grad()
    def predict(self, x):
        return self.forward(x)