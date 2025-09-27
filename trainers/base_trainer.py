import os, sys

cwd = s=os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import argparse
import importlib
import logging

import torch
import torch. nn as nn
from torch.utils.data import DataLoader

from ew_model import build_model, build_loss, build_optim
from dataset import build_dataset
from io import BytesIO

class TrainConfig:
    def __init__(self):
        self.path = None
        self.args = self.get_model_config()

    def get_model_config(self):
        parser = argparse.ArgumentParser(description="model config")
        parser.add_argument("--model_config_path", type=str, default="configs/two_tower_config", help="model config path")
        parser.add_argument("--worker_count", type=int, default=1, help="gpu worker count")
        #parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
        args = parser.parse_args()
        self.path = args.model_config_path
        if self.path.endswith("/"):
            self.path = self.path[:-1]
        return args


class BaseTrainer:
    def __init__(self):
        self.args = TrainConfig()
        self.model: nn.Module = None
        self.loss = None
        self.train_cfg = None
        self.device = 'cpu'
        self.build()
    
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
        # TODO: build dataset and optimizer and scheduler
        #TODO: move loss cfg into train
        self.train_cfg = args.get("train_cfg")
        self.device = self.train_cfg.get("device", "cpu")
        optim_cfg = self.train_cfg.get("optimizer")
        optim_args = dict(
            params=self.model.parameters(),
            **optim_cfg
        )
        self.loss = build_loss(loss)
        self.optim = build_optim(optim_args)
        self.model.to(self.device)
        self.dataset.to(self.device)

    def run(self):
        self.export_model()

    def save_model(self):
        buffer = BytesIO()
        torch.save(self.model.state_dict(), buffer)
        path = os.path.join("", "model.pth")
        with open(path, 'wb') as f:
            f.write(buffer.getvalue())

    def load_model(self, path='model.pth'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"model path {path} not exists")
        with open(path, 'rb') as f:
            byte = BytesIO(f.read())
            self.model.load_state_dict(torch.load(byte, weights_only=True))
    
    def export_model(self, path='model.onnx'):
        self.model.export(path)

    def model_test(self):
        datas = DataLoader(self.dataset, batch_size=1, shuffle=True)
        for i, (data, label) in enumerate(datas):
            print(data)
            out = self.model(data)
            print(out.shape)
            l = self.loss(out, label[:,0])
            print(l)
            self.optim.zero_grad()
            l.backward()
            self.optim.step()

    def rec_model_test(self):
        train_data = DataLoader(self.dataset, batch_size=1, shuffle=True)
        for i, (dummy, label) in enumerate(train_data):
            out = self.model(dummy)
            print(label)
            click = label[:, 0]
            loss = nn.functional.binary_cross_entropy(out, click)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            print(out.shape)
            print(loss)

if __name__ == "__main__":
    trainer = BaseTrainer()
    trainer.run()