# external_eval/train.py
"""
Linear Probe training module

Features:
1. Freeze backbone, train classification head only
2. Early stopping
3. LR scheduling
4. Save best model
"""

import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from configs import TRAIN_CONFIG, RESULTS_ROOT, DEVICE


def set_seed(seed: int):
    """Set random seed，ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: epochs to wait before stopping
            min_delta: minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Returns:
            True if this is the best score so far
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class LinearProbeTrainer:
    """
    Linear Probe trainer
    
    Freeze backbone, train classification head only
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        device: str = DEVICE,
        lr: float = TRAIN_CONFIG['lr'],
        weight_decay: float = TRAIN_CONFIG['weight_decay'],
        epochs: int = TRAIN_CONFIG['epochs'],
        patience: int = TRAIN_CONFIG['patience'],
        scheduler_type: str = TRAIN_CONFIG['scheduler'],
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.epochs = epochs
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer: only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # LR scheduler
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # Record
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.best_model_state = None
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validation"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Full training loop
        
        Returns:
            history: training history
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Linear Probe Training")
            print(f"{'='*60}")
            print(f"  Epochs: {self.epochs}")
            print(f"  Train samples: {len(self.train_loader.dataset)}")
            print(f"  Val samples: {len(self.val_loader.dataset)}")
            print(f"  Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # LR scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Early stopping check
            is_best = self.early_stopping(val_acc)
            if is_best:
                self.best_val_acc = val_acc
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                best_marker = " *" if is_best else ""
                print(f"Epoch [{epoch+1:3d}/{self.epochs}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%{best_marker} | "
                      f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            if self.early_stopping.early_stop:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training completed in {total_time:.1f}s")
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
            print(f"{'='*60}\n")
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def save_model(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }, path)
        print(f"Model saved to {path}")


def train_linear_probe(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    save_path: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function: train linear probe
    
    Args:
        model: model (backbone frozen)
        train_loader: training data
        val_loader: validation data
        num_classes: number of classes
        save_path: save path (optional)
        verbose: verbose output
        **kwargs: additional training params
    
    Returns:
        model: trained model
        history: training history
    """
    trainer = LinearProbeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        **kwargs
    )
    
    history = trainer.train(verbose=verbose)
    
    if save_path:
        trainer.save_model(save_path)
    
    return model, history


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Linear Probe Trainer test")
    print("=" * 60)
    
    from dataset import get_dataloader
    from models import create_model
    
    # Select a small dataset for testing
    dataset_name = 'glaucoma'
    model_name = 'imagenet_vit'
    
    print(f"\n[Test] Dataset: {dataset_name}, Model: {model_name}")
    
    # Load data
    train_loader = get_dataloader(dataset_name, 'train', batch_size=32)
    val_loader = get_dataloader(dataset_name, 'val', batch_size=32)
    
    # Create model
    from configs import DATASETS
    num_classes = DATASETS[dataset_name]['num_classes']
    model = create_model(model_name, num_classes=num_classes, freeze_backbone=True)
    
    # Training (5 epochs for testing)
    model, history = train_linear_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=5,  # For testing; use 50 for full training
        verbose=True
    )
    
    print("\n[Test] Training history:")
    print(f"  Final train acc: {history['train_acc'][-1]:.2f}%")
    print(f"  Final val acc: {history['val_acc'][-1]:.2f}%")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)