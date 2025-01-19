from common.register import Register
import torch.nn as nn

RECMODEL=Register("rec_models")
BACKBONE=Register("backbone")
LOSS=Register("loss")
PARSERS=Register("parsers")
NORMLIZATION=Register("normlization")
NORMLIZATION.register_module("BN", nn.BatchNorm2d, force=True)

HEAD=Register("head")
MODEL=Register("models")

def build_model(cfg):
    return MODEL.build(cfg)

def build_backbone(cfg):
    return BACKBONE.build(cfg)

def build_head(cfg):
    return HEAD.build(cfg)

def build_loss(cfg):
    return LOSS.build(cfg)
