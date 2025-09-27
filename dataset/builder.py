from torch.utils.data import DataLoader
from common.register import Register
from common.register import build_colate_fn_from_cfg

DATASET = Register("dataset")
COLATE_FN = Register("colate_fn", build_func=build_colate_fn_from_cfg)

def build_dataset(cfg):
    return DATASET.build(cfg)

def build_dataloader(cfg, dataset) -> DataLoader:
    copied_cfg = cfg.copy()
    num_record = dataset.num_record if hasattr(dataset, 'num_record') else 1
    batch_size = copied_cfg.get("batch_size", 1)
    assert batch_size % num_record == 0, f"batch size {batch_size} must be divisible by num_record {num_record}"
    real_batch_size = batch_size // num_record
    copied_cfg["batch_size"] = real_batch_size
    if "colate_fn" in copied_cfg:
        collate_fn = copied_cfg.pop("collate_fn")
        if isinstance(collate_fn, dict):
            collate_fn = COLATE_FN.build(collate_fn)
        return DataLoader(dataset, **copied_cfg, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, **copied_cfg)
