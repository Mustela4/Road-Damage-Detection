from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from vision.references.detection.utils import collate_fn
from transform_interface import get_geometry_transform, get_pixel_transform
from load_dataset import RARE_CLASSES

@dataclass
class DatasetConfig:
    annotations_file: str
    img_dir: str

class RoadDamageDataset(Dataset):
    def __init__(self, config: DatasetConfig, class_map: dict, transforms=None):
        super().__init__()
        self.img_dir = config.img_dir
        self.geometry_transforms = transforms
        self.pixel_transform = get_pixel_transform()
        self.class_to_int = class_map

        df = pd.read_csv(config.annotations_file)
        df.loc[df['class'].isin(RARE_CLASSES), 'class'] = 'Other'
        invalid = (df['xmin'] >= df['xmax']) | (df['ymin'] >= df['ymax'])
        df = df[~invalid].copy()

        self.df = df
        self.image_filenames = df['filename'].unique()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = str(Path(self.img_dir, image_filename))

        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann = self.df[self.df['filename'] == image_filename]
        boxes_np = ann[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = tv_tensors.BoundingBoxes(boxes_np, format="XYXY", canvas_size=image.shape[:2])

        labels_str = ann['class'].values
        if pd.isnull(labels_str).all():
            labels = torch.empty((0,), dtype=torch.int64)
            boxes = tv_tensors.BoundingBoxes(torch.empty((0, 4)), format="XYXY", canvas_size=image.shape[:2])
        else:
            labels = [self.class_to_int[lbl] for lbl in labels_str]
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        if boxes.shape[0] > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.empty((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((labels.shape[0],), dtype=torch.int64)

        if self.geometry_transforms:
            image, target = self.geometry_transforms(image, target)

        image = self.pixel_transform(image)
        return image, target

def get_dataloaders(train_csv, val_csv, train_img_dir, val_img_dir, class_to_int, batch_size: int):
    train_ds = RoadDamageDataset(
        DatasetConfig(train_csv, train_img_dir),
        class_to_int,
        transforms=get_geometry_transform(train=True),
    )
    val_ds = RoadDamageDataset(
        DatasetConfig(val_csv, val_img_dir),
        class_to_int,
        transforms=get_geometry_transform(train=False),
    )

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn= collate_fn)
    return train_loader, val_loader
