# external_eval/dataset.py
"""
MedIMeta dataset loading

Supports:
1. Read splits/train.txt, val.txt, test.txt
2. Load task_labels/*.npy labels
3. Read images/*.tiff images
4. Auto-convert grayscale to 3-channel
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from configs import DATASETS, DATA_ROOT, TRAIN_CONFIG


class MedIMetaDataset(Dataset):
    """
    MedIMeta dataset
    
    Args:
        dataset_name: dataset name (bus, fundus, glaucoma, mammo_calc, mammo_mass)
        split: split (train, val, test)
        transform: image transform
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__()
        
        assert dataset_name in DATASETS, f"Unknown dataset: {dataset_name}"
        assert split in ['train', 'val', 'test'], f"Unknown split: {split}"
        
        self.dataset_name = dataset_name
        self.split = split
        self.config = DATASETS[dataset_name]
        
        # Paths
        self.dataset_dir = os.path.join(DATA_ROOT, dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.split_file = os.path.join(self.dataset_dir, 'splits', f'{split}.txt')
        
        # Label file path
        task_name = self.config['task_name']
        self.label_file = os.path.join(self.dataset_dir, 'task_labels', f'{task_name}.npy')
        
        # Load data
        self.image_paths, self.image_indices = self._load_split()
        self.labels = self._load_labels()
        
        # Transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()
        
        print(f"[MedIMetaDataset] {dataset_name}/{split}: {len(self)} samples, "
              f"{self.config['num_classes']} classes")
    
    def _load_split(self) -> Tuple[List[str], List[int]]:
        """
        Load split file, get image paths and indices
        
        Returns:
            image_paths: list of full image paths
            image_indices: list of indices (for reading labels from npy)
        """
        image_paths = []
        image_indices = []
        
        with open(self.split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # line format: images/000006.tiff
                # Extract image number as index
                filename = os.path.basename(line)  # 000006.tiff
                idx = int(filename.split('.')[0])  # 6
                
                # Full path
                full_path = os.path.join(self.dataset_dir, line)
                
                image_paths.append(full_path)
                image_indices.append(idx)
        
        return image_paths, image_indices
    
    def _load_labels(self) -> np.ndarray:
        """
        Load label file
        
        Returns:
            labels: numpy array indexed by image number
        """
        labels = np.load(self.label_file)
        return labels
    
    def _default_transform(self) -> transforms.Compose:
        """
        Default image transforms
        
        Train: augmentation
        Val/test: normalize only
        """
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample
        
        Returns:
            image: [3, 224, 224] tensor
            label: integer label
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        # Convert grayscale to RGB (3-channel)
        if self.config['in_channels'] == 1:
            # Grayscale to RGB
            image = image.convert('RGB')
        else:
            # Ensure RGB
            image = image.convert('RGB')
        
        # Apply transform
        image = self.transform(image)
        
        # Get label
        image_idx = self.image_indices[idx]
        label = int(self.labels[image_idx])
        
        return image, label


def get_dataloader(
    dataset_name: str,
    split: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    transform: Optional[transforms.Compose] = None
) -> DataLoader:
    """
    Get DataLoader
    
    Args:
        dataset_name: dataset name
        split: data split
        batch_size: batch size (default from config)
        num_workers: num workers (default from config)
        shuffle: shuffle (default: True for train, False for val/test)
        transform: image transform
    
    Returns:
        DataLoader
    """
    # Default parameters
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']
    if num_workers is None:
        num_workers = TRAIN_CONFIG['num_workers']
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Create dataset
    dataset = MedIMetaDataset(
        dataset_name=dataset_name,
        split=split,
        transform=transform
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def get_all_dataloaders(
    dataset_name: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Get train, val, test DataLoaders
    
    Args:
        dataset_name: dataset name
        batch_size: batch size
        num_workers: num workers
    
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataloaders[split] = get_dataloader(
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers
        )
    
    return dataloaders


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MedIMetaDataset test")
    print("=" * 60)
    
    # Test each dataset
    for dataset_name in ['bus', 'fundus', 'glaucoma', 'mammo_calc', 'mammo_mass']:
        print(f"\n[Testing {dataset_name}]")
        
        try:
            # Create dataset
            dataset = MedIMetaDataset(dataset_name, split='train')
            
            # Get one sample
            image, label = dataset[0]
            print(f"  Image shape: {image.shape}")
            print(f"  Label: {label}")
            print(f"  Label range: {dataset.labels.min()} - {dataset.labels.max()}")
            
            # Create DataLoader
            dataloader = get_dataloader(dataset_name, split='train', batch_size=4)
            batch = next(iter(dataloader))
            print(f"  Batch image shape: {batch[0].shape}")
            print(f"  Batch label shape: {batch[1].shape}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)