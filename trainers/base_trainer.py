import os, sys

cwd = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(cwd, ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import argparse
import importlib
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ew_model import build_model, build_loss, build_optim
from dataset import build_dataset, build_dataloader, build_data_helper
from utils import parse_path
from io import BytesIO

from trainers.train_config import TrainConfig

class BaseTrainer:
    def __init__(self):
        self.args = TrainConfig()
        self.model: nn.Module = None
        self.loss = None
        self.train_cfg = None
        self.device = 'cpu'
        self.cfg = self.parse_model_args()
        self.build()
    
    def parse_model_args(self):
        model_config_path = self.args.model_config_path
        module_path = parse_path(model_config_path)
        logging.info(f"model config package: {module_path}")
        config = importlib.import_module(module_path)

        args = config.Compose()
        return args

    def build(self):
        args = self.cfg
        model = args.get('model_cfg')
        loss = args.get('loss_cfg')
        self.model = build_model(model)
        # TODO: build dataset and optimizer and scheduler
        #TODO: move loss cfg into train
        self.train_cfg = args.get("train_cfg")
        self.device = self.train_cfg.get("device", "cpu")
        optim_cfg = self.train_cfg.get("optimizer")
        optim_args = dict(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            **optim_cfg
        )
        self.loss = build_loss(loss)
        self.optim = build_optim(optim_args)

    def run(self):
        data_cfg = self.cfg.get("data_cfg")
        dataset_cfg = data_cfg.get("dataset_cfg")
        dataset = build_dataset(dataset_cfg)
        dataloader_cfg = data_cfg.get("dataloader_cfg")
        dataloader = build_dataloader(dataloader_cfg, dataset)
        datasetter_cfg = data_cfg.get("data_setter_cfg")
        datasetter = build_data_helper(datasetter_cfg)
        self.dataset = build_dataset(dataset_cfg)
        if self.args.mode in ("train", "restore"):
            self.train(self.model, dataloader, datasetter)

    def save_model(self, path='', fix=''):
        buffer = BytesIO()
        torch.save(self.model.state_dict(), buffer)
        path = os.path.join(path, f"model_weight_{fix}.pth")
        with open(path, 'wb') as f:
            f.write(buffer.getvalue())

    def load_model(self, path='', fix='final'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"model path {path} not exists")
        with open(os.path.join(path, f'model_weight_{fix}'), 'rb') as f:
            byte = BytesIO(f.read())
            self.model.load_state_dict(torch.load(byte, weights_only=True))
    
    def train(self, model, dataset, datasetter, **kwargs):
        model.to(self.device)
        model.train()
        print("training model...")
        for i, (data, label) in enumerate(dataset):
            data = datasetter(data, self.device)
            label = datasetter(label, self.device)
            out = model(data)
            loss = self.loss(out, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
    def export_model(self, path='model.onnx'):
        self.model.export(path)

    def model_test(self):
        datas = DataLoader(self.dataset, batch_size=1, shuffle=True)
        for i, (data, label) in enumerate(datas):
            print(data)
            out = self.model(data)
            print(out.shape)
            l = self.loss(out, label)
            print(l)
            self.optim.zero_grad()
            l.backward()
            self.optim.step()

    def rec_model_test(self):
        train_data = DataLoader(self.dataset, batch_size=1, shuffle=True)
        for i, (dummy, label) in enumerate(train_data):
            out = self.model(dummy)
            print(label)
            loss = self.loss(out, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            print(out.shape)
            print(loss)

if __name__ == "__main__":
    trainer = BaseTrainer()
    trainer.run()