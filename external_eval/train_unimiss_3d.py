"""
UniMiSS 3D External Validation Training Script - Enhanced

Added:
- TTA (Test Time Augmentation)
- Warmup scheduler
- More epochs support
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from dataset_3d import MosMedDataset3D


def create_model(model_name, num_classes, weights_path=None):
    """Create model"""
    if model_name == 'unimiss':
        from models_unimiss import create_unimiss_classifier
        model = create_unimiss_classifier(
            num_classes=num_classes,
            pretrained_path=weights_path or 'weights/self_supervised_unimiss_nnunet_small_5022.pth',
            in_chans_2d=3,
            in_chans_3d=1
        )
    elif model_name.startswith('med3d'):
        from models_3d import create_model_3d
        model = create_model_3d(model_name, num_classes=num_classes, freeze_backbone=False)
    elif model_name == 'imagenet_vit':
        from models_3d import create_model_3d
        model = create_model_3d('imagenet_vit_3d', num_classes=num_classes, freeze_backbone=False)
    elif model_name == 'ours':
        from models_3d import create_model_3d
        model = create_model_3d('ours_expert_c', num_classes=num_classes, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device, use_tta=False):
    """Evaluate with TTA support"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_tta:
                # TTA: original + 3 flips, average
                outputs_list = []
                
                # Original
                outputs_list.append(model(images))
                
                # Flip along D axis
                outputs_list.append(model(images.flip(2)))
                
                # Flip along H axis
                outputs_list.append(model(images.flip(3)))
                
                # Flip along W axis
                outputs_list.append(model(images.flip(4)))
                
                # Average
                outputs = torch.stack(outputs_list).mean(0)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100. * correct / total
    avg_loss = total_loss / total
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except:
        auc = 50.0
    
    try:
        ap = average_precision_score(all_labels, all_probs) * 100
    except:
        ap = 50.0
    
    f1 = f1_score(all_labels, all_preds, average='binary') * 100
    
    return avg_loss, acc, auc, ap, f1


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.base_lrs[i] * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.min_lr + 0.5 * (self.base_lrs[i] - self.min_lr) * (1 + np.cos(np.pi * progress))
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ours',
                       choices=['unimiss', 'imagenet_vit', 'med3d_resnet18', 'med3d_resnet34', 'med3d_resnet50', 'ours'])
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='results_3d_ours_v3')
    parser.add_argument('--use_tta', action='store_true', help='Use Test Time Augmentation')
    parser.add_argument('--freeze_backbone', action='store_true')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"TTA: {args.use_tta}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    
    # Create datasets
    train_dataset = MosMedDataset3D(split='train', augment=True)
    val_dataset = MosMedDataset3D(split='val', augment=False)
    test_dataset = MosMedDataset3D(split='test', augment=False)
    
    # Class weights
    train_labels = train_dataset.labels.astype(int)
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class distribution (train): {class_counts}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Dataset: COVID-CT (MosMedData)")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    num_classes = 2
    model = create_model(args.model, num_classes, args.weights)
    model = model.to(device)
    
    # Freeze backbone if specified
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("Backbone frozen (Linear Probe mode)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
    
    # Optimizer with different lr for backbone and head
    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head, weight_decay=args.weight_decay
        )
    else:
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if 'fc' in n and p.requires_grad]
        
        if backbone_params and head_params:
            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': args.lr},
                {'params': head_params, 'lr': args.lr_head}
            ], weight_decay=args.weight_decay)
            print(f"Full Fine-tuning lr: backbone={args.lr}, head={args.lr_head}")
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr_head, weight_decay=args.weight_decay
            )
    
    # Warmup + Cosine scheduler
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    best_state_dict = None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_ap, val_f1 = evaluate(model, val_loader, criterion, device, use_tta=False)
        
        lr = scheduler.get_last_lr()[0]
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Acc: {val_acc:.2f}%, AUC: {val_auc:.2f}% | lr: {lr:.2e}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Final evaluation with TTA
    print("\n" + "=" * 50)
    print("Final evaluation on test set...")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    # Use TTA if specified
    test_loss, test_acc, test_auc, test_ap, test_f1 = evaluate(
        model, test_loader, criterion, device, use_tta=args.use_tta
    )
    
    print(f"Test Results (TTA={args.use_tta}): ACC: {test_acc:.2f}%, AUC: {test_auc:.2f}%, AP: {test_ap:.2f}%, F1: {test_f1:.2f}%")
    
    # Save results
    results = {
        'model': args.model,
        'dataset': 'covid_ct',
        'seed': args.seed,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'test_f1': test_f1,
        'use_tta': args.use_tta,
        'args': vars(args)
    }
    
    result_file = os.path.join(args.output_dir, f'{args.model}_covid_seed{args.seed}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    main()