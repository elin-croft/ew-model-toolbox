import argparse
import importlib
import logging

import torch
import torch. nn as nn

from ew_model import build_model, build_loss
from dataset import build_dataset

class TrainConfig:
    def __init__(self):
        self.path = None
        self.get_model_config()

    def get_model_config(self):
        parser = argparse.ArgumentParser(description="model config")
        parser.add_argument("--model_config_path", type=str, default=None, help="model config path")
        args = parser.parse_args()
        self.path = args.model_config_path
        if self.path.endswith("/"):
            self.path = self.path[:-1]


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
        logging.info(f"model config package: {module_path}")
        config = importlib.import_module(module_path)

        args = config.Compose()
        return args

    def build(self):
        args = self.parse_model_args()
        model = args.get('model_cfg')
        dataset_cfg = args.get("dataset_cfg")
        loss = args.get('loss_cfg')
        self.model = build_model(model)
        self.dataset = build_dataset(dataset_cfg)
        self.loss = build_loss(loss)
        # TODO: build dataset and optimizer and scheduler
        self.train_cfg = args.get("train_cfg")
        self.device = self.train_cfg.get("device", "cpu")
        self.model.to(self.device)
        self.dataset.to(self.device)

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

    def rec_model_test(self):
        self.model.eval()
        for i, (dummy, label) in enumerate(self.dataset):
            out = self.model(dummy)
            print(out.shape)