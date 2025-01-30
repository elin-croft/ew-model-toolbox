import argparse
import importlib
import torch
import torch. nn as nn
from ew_model import build_model, build_loss

class TrainConfig:
    def __init__(self):
        self.path = None
        self.get_model_config()

    def get_model_config(self):
        parser = argparse.ArgumentParser(description="model config")
        parser.add_argument("--model_config_path", type=str, default=None, help="model config path")
        args = parser.parse_args()
        self.path = args.model_config_path


class BaseTrainer:
    def __init__(self):
        self.args = TrainConfig()
        self.model: nn.Module = None
        self.loss = None
        self.train_cfg = None
        self.device = 'cpu'
    
    def parse_model_args(self):
        self.model_config_path = self.args.path
        module_path = self.model_config_path.replace("/", ".").replace(".py", "")
        print(module_path)
        config = importlib.import_module(module_path)
        args = config.compose()
        return args

    def build(self):
        args = self.parse_model_args()
        model = args.get('model')
        loss = args.get('loss')
        self.model = build_model(model)
        self.loss = build_loss(loss)
        self.train_cfg = args.get("train_cfg")
        self.device = self.train_cfg.get("device", "cpu")
        self.model.to(self.device)

    def train_step(self):
        pass

    def val_step(self):
        pass

    def test_step(self):
        pass

    def train(self):
        pass

    def val(self):
        pass

    def test(self):
        pass

    def model_test(self):
        dummy = torch.randn((10,3,224,224)).to(self.device)
        label = torch.randint(0,1000,(10,)).to(self.device)
        out = self.model(dummy)
        print(out.shape)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        l = self.loss(out, label)
        print(l)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()