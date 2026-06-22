# dataloader/medmnist_loader.py
"""
MedMNIST data loading.

Uses the official medmnist library; supports:
    - 2D (224x224) and 3D (64x64x64) datasets
    - Dual-augmentation mode for Teacher-Student training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

import medmnist
from medmnist import INFO

from config.datasets import DATASET_CONFIGS, get_dataset_config
from .transforms import DualTransform2D, DualTransform3D, get_transforms


class MedMNISTDataset(Dataset):
    """
    Wrapper around the official MedMNIST dataset.

    Args:
        dataset_name:   E.g. 'PathMNIST', 'OrganMNIST3D'.
        split:          'train', 'val', or 'test'.
        data_root:      Root directory for data files.
        dual_transform: If True, return two augmented views (train only).
    """

    def __init__(self, dataset_name, split="train", data_root="./data",
                 dual_transform=False, transform=None):
        self.dataset_name = dataset_name
        self.split = split
        self.dual_transform = dual_transform
        self.config = get_dataset_config(dataset_name)
        self.is_3d = self.config.is_3d

        # Load via medmnist library
        info = INFO[self.config.medmnist_name]
        DataClass = getattr(medmnist, info["python_class"])
        kwargs = dict(split=split, transform=None, download=False, root=data_root)
        if self.is_3d:
            kwargs["size"] = 64
        else:
            kwargs["size"] = 224
            kwargs["as_rgb"] = True
        self.dataset = DataClass(**kwargs)

        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif dual_transform and split == "train":
            self.transform = DualTransform3D() if self.is_3d else DualTransform2D()
        else:
            self.transform = get_transforms(self.is_3d, split, dual=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        label = torch.tensor(label).squeeze()
        if self.dual_transform and self.split == "train":
            img_s, img_t = self.transform(img)
            return (img_s, img_t), label
        return self.transform(img), label


def create_dataloader(dataset_name, split="train", data_root="./data",
                      batch_size=32, num_workers=4, dual_transform=False,
                      shuffle=None, pin_memory=True):
    ds = MedMNISTDataset(dataset_name, split, data_root, dual_transform)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      drop_last=(split == "train"))


def create_all_dataloaders(dataset_list, data_root="./data", batch_size=32,
                           num_workers=4, dual_transform=False):
    loaders = {}
    for name in dataset_list:
        loaders[name] = {
            s: create_dataloader(name, s, data_root, batch_size, num_workers,
                                 dual_transform if s == "train" else False)
            for s in ("train", "val", "test")
        }
        tr = loaders[name]["train"].dataset
        print(f"  {name}: train={len(tr)}, "
              f"val={len(loaders[name]['val'].dataset)}, "
              f"test={len(loaders[name]['test'].dataset)}")
    return loaders