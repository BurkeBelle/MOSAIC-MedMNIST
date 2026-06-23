# external_eval/evaluate.py
"""
Evaluation module

Computes metrics:
1. ACC (Accuracy)
2. AUC (Area Under ROC Curve)
3. F1 (F1 Score)
4. AP (Average Precision)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from configs import DEVICE, DATASETS


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = DEVICE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions
    
    Args:
        model: model
        dataloader: dataloader
        device: device
    
    Returns:
        y_true: true labels [N]
        y_pred: predicted labels [N]
        y_prob: prediction probabilities [N, C]
    """
    model.eval()
    model = model.to(device)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    for images, labels in dataloader:
        images = images.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    return y_true, y_pred, y_prob


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        y_true: true labels [N]
        y_pred: predicted labels [N]
        y_prob: prediction probabilities [N, C]
        num_classes: number of classes
    
    Returns:
        metrics: dict with ACC, AUC, F1, AP
    """
    metrics = {}
    
    # 1. Accuracy
    metrics['ACC'] = accuracy_score(y_true, y_pred) * 100  # convert to percentage
    
    # 2. AUC (Area Under ROC Curve)
    try:
        if num_classes == 2:
            # Binary: use positive class probability
            metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1]) * 100
        else:
            # Multi-class: one-vs-rest
            metrics['AUC'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            ) * 100
    except Exception as e:
        print(f"  Warning: AUC calculation failed: {e}")
        metrics['AUC'] = 0.0
    
    # 3. F1 Score
    if num_classes == 2:
        # Binary
        metrics['F1'] = f1_score(y_true, y_pred, average='binary') * 100
    else:
        # Multi-class: macro average
        metrics['F1'] = f1_score(y_true, y_pred, average='macro') * 100
    
    # 4. Average Precision (AP)
    try:
        if num_classes == 2:
            # Binary: use positive class probability
            metrics['AP'] = average_precision_score(y_true, y_prob[:, 1]) * 100
        else:
            # Multi-class: one-hot encoding
            y_true_onehot = np.eye(num_classes)[y_true]
            metrics['AP'] = average_precision_score(
                y_true_onehot, y_prob, average='macro'
            ) * 100
    except Exception as e:
        print(f"  Warning: AP calculation failed: {e}")
        metrics['AP'] = 0.0
    
    return metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    device: str = DEVICE,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model
    
    Args:
        model: model
        test_loader: test dataloader
        num_classes: number of classes
        device: device
        verbose: verbose output
    
    Returns:
        metrics: evaluation metrics dict
    """
    if verbose:
        print(f"\n[Evaluating] {len(test_loader.dataset)} samples, {num_classes} classes")
    
    # Get predictions
    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    
    if verbose:
        print(f"  ACC: {metrics['ACC']:.2f}%")
        print(f"  AUC: {metrics['AUC']:.2f}%")
        print(f"  F1:  {metrics['F1']:.2f}%")
        print(f"  AP:  {metrics['AP']:.2f}%")
    
    return metrics


def evaluate_and_report(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    class_names: Optional[Dict[int, str]] = None,
    device: str = DEVICE
) -> Dict:
    """
    Evaluate model and generate detailed report
    
    Args:
        model: model
        test_loader: test dataloader
        num_classes: number of classes
        class_names: class name dict
        device: device
    
    Returns:
        report: dict with metrics and detailed report
    """
    print(f"\n{'='*60}")
    print("Evaluation Report")
    print(f"{'='*60}")
    
    # Get predictions
    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    
    print(f"\n[Main Metrics]")
    print(f"  ACC: {metrics['ACC']:.2f}%")
    print(f"  AUC: {metrics['AUC']:.2f}%")
    print(f"  F1:  {metrics['F1']:.2f}%")
    print(f"  AP:  {metrics['AP']:.2f}%")
    
    # Confusion matrix
    print(f"\n[Confusion Matrix]")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Classification report
    print(f"\n[Classification Report]")
    if class_names:
        target_names = [class_names[i] for i in range(num_classes)]
    else:
        target_names = [f"Class {i}" for i in range(num_classes)]
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    
    print(f"{'='*60}\n")
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def format_metrics(metrics: Dict[str, float], decimal: int = 2) -> str:
    """
    Format metrics as string
    
    Args:
        metrics: metrics dict
        decimal: decimal places
    
    Returns:
        formatted string
    """
    return (f"ACC: {metrics['ACC']:.{decimal}f}% | "
            f"AUC: {metrics['AUC']:.{decimal}f}% | "
            f"F1: {metrics['F1']:.{decimal}f}% | "
            f"AP: {metrics['AP']:.{decimal}f}%")


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Evaluate module test")
    print("=" * 60)
    
    from dataset import get_dataloader
    from models import create_model
    from train import train_linear_probe
    
    # Select a small dataset for testing
    dataset_name = 'glaucoma'
    model_name = 'imagenet_vit'
    
    print(f"\n[Test] Dataset: {dataset_name}, Model: {model_name}")
    
    # Load data
    train_loader = get_dataloader(dataset_name, 'train', batch_size=32)
    val_loader = get_dataloader(dataset_name, 'val', batch_size=32)
    test_loader = get_dataloader(dataset_name, 'test', batch_size=32)
    
    # Create model
    num_classes = DATASETS[dataset_name]['num_classes']
    class_names = DATASETS[dataset_name]['labels']
    
    model = create_model(model_name, num_classes=num_classes, freeze_backbone=True)
    
    # Training (5 epochs for testing)
    print("\n[Training]")
    model, history = train_linear_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=5,
        verbose=True
    )
    
    # Evaluate
    print("\n[Testing on test set]")
    metrics = evaluate_model(model, test_loader, num_classes)
    
    # Detailed report
    report = evaluate_and_report(
        model, test_loader, num_classes, class_names=class_names
    )
    
    print("\n[Summary]")
    print(f"  {format_metrics(metrics)}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)