# external_eval/dataset_3d.py
"""
MosMedData 3D data loading

Data format:
  train.npy: (888, 64, 64, 64) float32
  test.npy:  (222, 64, 64, 64) float32
  train_labels.npy: (888,) int
  test_labels.npy:  (222,) int

Output:
  volume: (1, 64, 64, 64) float32  (normalized to [0,1])
  label:  int (0=Normal, 1=COVID)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs_3d import DATASET_3D, TRAIN_CONFIG


class MosMedDataset3D(Dataset):
    """MosMedData COVID-19 CT 3D dataset"""

    def __init__(self, split='train', augment=False):
        """
        Args:
            split: 'train' or 'test'
            augment: apply augmentation (train only)
        """
        cfg = DATASET_3D

        if split == 'train':
            self.volumes = np.load(cfg['train_file'])
            self.labels = np.load(cfg['train_labels_file'])
        elif split == 'test':
            self.volumes = np.load(cfg['test_file'])
            self.labels = np.load(cfg['test_labels_file'])
        elif split == 'val':
            # Split 20% from training as validation
            all_volumes = np.load(cfg['train_file'])
            all_labels = np.load(cfg['train_labels_file'])
            n = len(all_volumes)
            n_val = int(n * 0.2)
            # Fixed split: last 20% as validation
            self.volumes = all_volumes[n - n_val:]
            self.labels = all_labels[n - n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.augment = augment and (split == 'train')

        # Normalize to [0, 1]
        v_min = self.volumes.min()
        v_max = self.volumes.max()
        if v_max > v_min:
            self.volumes = (self.volumes - v_min) / (v_max - v_min)
        self.volumes = self.volumes.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        volume = self.volumes[idx]     # (64, 64, 64)
        label = int(self.labels[idx])

        # Data augmentation
        if self.augment:
            volume = self._augment(volume)

        # Add channel dim: (64,64,64) -> (1, 64, 64, 64)
        volume = torch.from_numpy(volume).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return volume, label

    def _augment(self, volume):
        """Simple 3D augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=2).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=1).copy()

        # Random depth flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=0).copy()

        # Random intensity shift (±5%)
        if np.random.random() > 0.5:
            shift = np.random.uniform(-0.05, 0.05)
            volume = np.clip(volume + shift, 0, 1)

        # Random intensity scaling (0.95~1.05)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.95, 1.05)
            volume = np.clip(volume * scale, 0, 1)

        return volume


def get_dataloader_3d(split='train', batch_size=None, shuffle=None):
    """
    Get DataLoader

    Args:
        split: 'train', 'val', 'test'
        batch_size: batch size (default from config)
        shuffle: shuffle (default: True for train)
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']
    if shuffle is None:
        shuffle = (split == 'train')

    augment = (split == 'train')
    dataset = MosMedDataset3D(split=split, augment=augment)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    return loader


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MosMedData 3D Dataset Test")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        dataset = MosMedDataset3D(split=split)
        print(f"\n[{split}]")
        print(f"  Samples: {len(dataset)}")
        print(f"  Labels distribution: {np.bincount(dataset.labels.astype(int))}")

        vol, lab = dataset[0]
        print(f"  Volume shape: {vol.shape}")
        print(f"  Volume range: [{vol.min():.4f}, {vol.max():.4f}]")
        print(f"  Label: {lab.item()}")

    print("\n--- DataLoader Test ---")
    loader = get_dataloader_3d('train', batch_size=4)
    batch_vol, batch_lab = next(iter(loader))
    print(f"Batch volume: {batch_vol.shape}")
    print(f"Batch labels: {batch_lab}")
    print(f"Batch dtype: {batch_vol.dtype}")