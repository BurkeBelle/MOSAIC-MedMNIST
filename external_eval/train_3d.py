"""
3D Linear Probe training

Freeze backbone, train classification head only
Supports: early stopping, cosine LR scheduler
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from configs_3d import TRAIN_CONFIG, RESULTS_ROOT, DEVICE


def set_seed(seed):
    """Set random seed，ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_linear_probe(
    model,
    train_loader,
    val_loader,
    num_classes=2,
    epochs=None,
    verbose=True,
    finetune=False,
    backbone_lr=None,
    adapter_lr=None,
) -> Tuple[nn.Module, dict]:
    """
    Linear Probe / Full fine-tuning training

    Args:
        model: model (backbone frozen or unfrozen)
        train_loader: train DataLoader
        val_loader: validation DataLoader
        num_classes: number of classes
        epochs: number of epochs
        verbose: print training progress
        finetune: True = full fine-tuning (backbone uses small LR)
        backbone_lr: backbone LR for fine-tuning (None=auto)
        adapter_lr: adapter LR for fine-tuning (None=same as backbone)

    Returns:
        model: trained model
        history: training history {'train_loss': [], 'val_loss': [], 'val_acc': []}
    """
    if epochs is None:
        epochs = TRAIN_CONFIG['epochs']

    # Compute class weights (handle class imbalance)
    train_labels = train_loader.dataset.labels.astype(int)
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    if verbose:
        print(f"  Class distribution: {class_counts}")
        print(f"  Class weights: {class_weights.cpu().numpy()}")
    # ==========================================================

    # Optimizer
    if finetune:
        # Fine-tuning: parameter groups
        backbone_params = []
        adapter_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # External head: 'fc.weight' / 'fc.bias'
                if name.startswith('fc.'):
                    head_params.append(param)
                # Adapter-related params
                elif 'adapter' in name:
                    adapter_params.append(param)
                else:
                    backbone_params.append(param)

        if backbone_lr is None:
            backbone_lr = TRAIN_CONFIG['lr'] * 0.01  # default 1e-5
        head_lr = TRAIN_CONFIG['lr']                  # 1e-3

        # Build parameter groups
        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr})
        if adapter_params and adapter_lr is not None:
            param_groups.append({'params': adapter_params, 'lr': adapter_lr})
        elif adapter_params:
            # No separate adapter LR, group with backbone
            param_groups[0]['params'] = backbone_params + adapter_params
        if head_params:
            param_groups.append({'params': head_params, 'lr': head_lr})

        optimizer = optim.Adam(param_groups, weight_decay=TRAIN_CONFIG['weight_decay'])

        if verbose:
            if adapter_lr is not None and adapter_params:
                print(f"  3-tier fine-tuning:")
                print(f"    backbone_lr={backbone_lr:.1e} ({sum(p.numel() for p in backbone_params):,} params)")
                print(f"    adapter_lr ={adapter_lr:.1e} ({sum(p.numel() for p in adapter_params):,} params)")
                print(f"    head_lr    ={head_lr:.1e} ({sum(p.numel() for p in head_params):,} params)")
            else:
                total_bk = backbone_params + adapter_params
                print(f"  2-tier fine-tuning:")
                print(f"    backbone_lr={backbone_lr:.1e} ({sum(p.numel() for p in total_bk):,} params)")
                print(f"    head_lr    ={head_lr:.1e} ({sum(p.numel() for p in head_params):,} params)")
    else:
        # Linear probe: only optimize trainable params (fc head)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            trainable_params,
            lr=TRAIN_CONFIG['lr'],
            weight_decay=TRAIN_CONFIG['weight_decay'],
        )

    # LR scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Use weighted cross-entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # =================================================

    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    patience = TRAIN_CONFIG['patience']
    best_state_dict = None

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
    }

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch_vol, batch_lab in train_loader:
            batch_vol = batch_vol.to(DEVICE)
            batch_lab = batch_lab.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_vol)
            loss = criterion(outputs, batch_lab)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_vol.size(0)
            n_train += batch_vol.size(0)

        train_loss /= n_train

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        n_val = 0

        with torch.no_grad():
            for batch_vol, batch_lab in val_loader:
                batch_vol = batch_vol.to(DEVICE)
                batch_lab = batch_lab.to(DEVICE)

                outputs = model(batch_vol)
                loss = criterion(outputs, batch_lab)

                val_loss += loss.item() * batch_vol.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_lab).sum().item()
                n_val += batch_vol.size(0)

        val_loss /= n_val
        val_acc = val_correct / n_val

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # LR update
        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or patience_counter == 0):
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"val_acc={val_acc:.4f}, "
                  f"best={best_val_acc:.4f}, "
                  f"lr={lr:.6f}"
                  f"{'  *' if patience_counter == 0 else ''}")

        if TRAIN_CONFIG['early_stopping'] and patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(patience={patience}, best_val_acc={best_val_acc:.4f})")
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history