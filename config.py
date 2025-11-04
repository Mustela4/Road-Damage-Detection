from dataclasses import dataclass, field
from pathlib import Path
import torch
import os
import random
import numpy as np

@dataclass
class Config:
    seed: int = 42
    base_path: str = "data/processed"
    train_csv: str = field(init=False)
    val_csv: str = field(init=False)
    train_img_dir: str = field(init=False)
    val_img_dir: str = field(init=False)
    
    num_epochs: int = 10
    batch_size: int = 8
    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 5e-4
    step_lr_gamma: float = 0.1
    log_dir: str = "runs/rdd_detection_v1"
    ckpt_dir: str = "checkpoints"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.base_path = str(Path(self.base_path).resolve())
        self.train_csv = str(Path(self.base_path, "train_annotations.csv"))
        self.val_csv   = str(Path(self.base_path, "val_annotations.csv"))
        self.train_img_dir = str(Path(self.base_path, "train/images"))
        self.val_img_dir   = str(Path(self.base_path, "val/images"))

        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch = __import__("torch")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
