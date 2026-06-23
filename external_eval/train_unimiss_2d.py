"""
UniMiSS 2D External Validation Training Script

For 2D external validation: BUS, Fundus, Glaucoma, Mammo-Calc, Mammo-Mass
Uses existing dataset.py (MedIMetaDataset)
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from datetime import datetime

from models_unimiss import create_unimiss_classifier
from dataset import MedIMetaDataset
from configs import DATASETS


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


def evaluate(model, loader, criterion, device, num_classes=2):
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100. * correct / total
    avg_loss = total_loss / total
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    if num_classes == 2:
        # Binary classification
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1]) * 100
        except:
            auc = 50.0
        try:
            ap = average_precision_score(all_labels, all_probs[:, 1]) * 100
        except:
            ap = 50.0
    else:
        # Multi-class: use macro average
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
        except:
            auc = 50.0
        ap = 0.0  # AP not well-defined for multi-class
    
    # F1 score
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    return avg_loss, acc, auc, ap, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['bus', 'fundus', 'glaucoma', 'mammo_calc', 'mammo_mass'])
    parser.add_argument('--weights', type=str, 
                       default='weights/self_supervised_unimiss_nnunet_small_5022.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='results_unimiss_2d')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone, only train fc head')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get number of classes from config
    num_classes = DATASETS[args.dataset]['num_classes']
    print(f"Dataset: {args.dataset}, Classes: {num_classes}")
    
    # Create datasets using MedIMetaDataset
    train_dataset = MedIMetaDataset(dataset_name=args.dataset, split='train')
    val_dataset = MedIMetaDataset(dataset_name=args.dataset, split='val')
    test_dataset = MedIMetaDataset(dataset_name=args.dataset, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model with correct num_classes
    model = create_unimiss_classifier(
        num_classes=num_classes,
        pretrained_path=args.weights,
        in_chans_2d=3,
        in_chans_3d=1
    )
    model = model.to(device)
    
    # Freeze backbone if specified
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("Backbone frozen, only training fc head")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
    
    # Optimizer with different lr for backbone and head
    if not args.freeze_backbone:
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if 'fc' in n and p.requires_grad]
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr},
            {'params': head_params, 'lr': args.lr_head}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head, weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop - use validation for model selection
    best_val_auc = 0
    best_epoch = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_ap, val_f1 = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.2f}%, F1: {val_f1:.2f}%")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            # Save best model
            torch.save(model.state_dict(), 
                      os.path.join(args.output_dir, f'unimiss_{args.dataset}_seed{args.seed}_best.pth'))
    
    # Final evaluation on test set (only once at the end)
    print("\n" + "=" * 50)
    print("Final evaluation on test set...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'unimiss_{args.dataset}_seed{args.seed}_best.pth')))
    test_loss, test_acc, test_auc, test_ap, test_f1 = evaluate(model, test_loader, criterion, device, num_classes)
    
    print(f"Test Results: ACC: {test_acc:.2f}%, AUC: {test_auc:.2f}%, AP: {test_ap:.2f}%, F1: {test_f1:.2f}%")
    
    # Save final results
    results = {
        'dataset': args.dataset,
        'seed': args.seed,
        'num_classes': num_classes,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'test_f1': test_f1,
        'args': vars(args)
    }
    
    result_file = os.path.join(args.output_dir, f'unimiss_{args.dataset}_seed{args.seed}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    main()
    
if __name__ == '__main__':
    main()