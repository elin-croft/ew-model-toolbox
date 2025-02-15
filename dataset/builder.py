from common.register import Register

DATASET = Register("dataset")

def build_dataset(cfg):
    return DATASET.build(cfg)