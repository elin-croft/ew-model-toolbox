import os, sys
cwd = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(cwd, ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
import argparse
from utils import str2bool

class Config:
    def __str__(self):
        format_string = self.__class__.__name__ + "(\n"
        for k, v in vars(self).items():
            format_string += f"   {k}={v}"
            format_string += "\n"
        return format_string + ")"

class TrainConfig(Config):
    def __init__(self):
        self.path = None
        args = self.get_model_config()
        self.model_config_path = args.model_config_path
        self.worker_count = args.worker_count
        self.restore_path = args.restore_path
        self.checkpoint_path = args.checkpoint_path
        self.compile_model = args.compile_model
        self.mode = args.mode
        print(self)

    def get_model_config(self):
        parser = argparse.ArgumentParser(description="model config")
        parser.add_argument("--model_config_path", type=str, default="configs/two_tower_config", help="model config path")
        parser.add_argument("--worker_count", type=int, default=1, help="gpu worker count")
        parser.add_argument("--restore_path", type=str, default="", help="restore checkpoint path")
        parser.add_argument("--checkpoint_path", type=str, default="./", help="checkpoint path")
        parser.add_argument("--compile_model", type=str2bool, default=False, help="whether to compile model")
        parser.add_argument("--mode", type=str, default="train", help="train or restore or test or export")
        parser.add_argument("--show_config", type=str2bool, default=False, help="whether to show config")
        args, _ = parser.parse_known_args()
        for k, v in vars(args).items():
            self.__setattr__(k, v)
        return args
