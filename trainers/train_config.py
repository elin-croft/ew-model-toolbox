import argparse

class TrainConfig:
    def __init__(self):
        self.path = None
        args = self.get_model_config()
        self.model_config_path = args.model_config_path
        self.worker_count = args.worker_count
        self.checkpoint_path = args.checkpoint_path
        self.mode = args.mode

    def get_model_config(self):
        parser = argparse.ArgumentParser(description="model config")
        parser.add_argument("--model_config_path", type=str, default="configs/two_tower_config", help="model config path")
        parser.add_argument("--worker_count", type=int, default=1, help="gpu worker count")
        parser.add_argument("--restore_path", type=str, default="", help="restore checkpoint path")
        parser.add_argument("--checkpoint_path", type=str, default="./", help="checkpoint path")
        parser.add_argument("--mode", type=str, default="train", help="train or restore or test or export")
        args = parser.parse_args()
        for k, v in vars(args).items():
            print(f"{k}: {v}")
            self.__setattr__(k, v)
        return args
