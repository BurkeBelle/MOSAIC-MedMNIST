# dataloader/transforms.py
"""
2D / 3D data augmentation.

Key design choices:
    - 3D: intensity-only augmentation (no spatial flips/rotations),
      because medical 3D volumes have fixed anatomical orientations.
    - 2D: standard ImageNet-style augmentation.
    - Dual-transform wrappers produce two views for Teacher–Student training.
"""

import random
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple


# ======================================================================
# 2D Transforms
# ======================================================================

class Transforms2D:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    @classmethod
    def get_train_transform(cls, for_teacher=False):
        normalize = transforms.Normalize(mean=cls.MEAN, std=cls.STD)
        if for_teacher:
            return transforms.Compose([transforms.ToTensor(), normalize])
        return transforms.Compose([
            transforms.ToTensor(), normalize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])

    @classmethod
    def get_eval_transform(cls):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.MEAN, std=cls.STD),
        ])


# ======================================================================
# 3D Transforms (intensity only — no spatial augmentation)
# ======================================================================

class Normalize3D:
    """Normalize 3D volume to [0, 1] and ensure channel dim."""
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.float()
        if x.max() > 1:
            x = x / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x


class RandomIntensity3D:
    """Random brightness and contrast shift (no spatial change)."""
    def __init__(self, brightness=(-0.1, 0.1), contrast=(0.9, 1.1), p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = x + random.uniform(*self.brightness)
        if random.random() < self.p:
            c = random.uniform(*self.contrast)
            x = (x - x.mean()) * c + x.mean()
        return torch.clamp(x, 0, 1)


class Compose3D:
    def __init__(self, ts):
        self.transforms = ts
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Transforms3D:
    @classmethod
    def get_train_transform(cls, for_teacher=False):
        if for_teacher:
            return Normalize3D()
        return Compose3D([Normalize3D(), RandomIntensity3D()])

    @classmethod
    def get_eval_transform(cls):
        return Normalize3D()


# ======================================================================
# Dual-transform wrappers (Teacher–Student)
# ======================================================================

class DualTransform2D:
    def __init__(self):
        self.student = Transforms2D.get_train_transform(for_teacher=False)
        self.teacher = Transforms2D.get_train_transform(for_teacher=True)
    def __call__(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.student(img), self.teacher(img)


class DualTransform3D:
    def __init__(self):
        self.student = Transforms3D.get_train_transform(for_teacher=False)
        self.teacher = Transforms3D.get_train_transform(for_teacher=True)
    def __call__(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.student(img), self.teacher(img)


def get_transforms(is_3d, split="train", dual=False):
    if split == "train":
        if dual:
            return DualTransform3D() if is_3d else DualTransform2D()
        T = Transforms3D if is_3d else Transforms2D
        return T.get_train_transform(for_teacher=False)
    T = Transforms3D if is_3d else Transforms2D
    return T.get_eval_transform()