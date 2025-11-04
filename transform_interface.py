from torchvision.transforms import v2 as Tv2
from PIL import Image
import numpy as np
import torch

class ConvertImageToNumpy(object):
    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return image, target

def get_geometry_transform(train: bool):
    transforms = [ConvertImageToNumpy()]
    if train:
        transforms.append(Tv2.RandomHorizontalFlip(p=0.5))
    return Tv2.Compose(transforms)

def get_pixel_transform():
    transforms = [
        Tv2.ToImage(),  
        Tv2.ToDtype(torch.float32, scale=True),
        Tv2.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ]
    return Tv2.Compose(transforms)
