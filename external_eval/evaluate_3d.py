# external_eval/evaluate_3d.py
"""
3D model evaluation

Metrics (same as 2D):
1. ACC (Accuracy)
2. AUC (Area Under ROC Curve)
3. F1 (F1 Score)
4. AP (Average Precision)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

from configs_3d import DEVICE


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = DEVICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions

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

    for volumes, labels in dataloader:
        volumes = volumes.to(device)

        outputs = model(volumes)
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
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute evaluation metrics (same as 2D evaluate.py)

    Returns:
        metrics: {'ACC': float, 'AUC': float, 'F1': float, 'AP': float}
                 All values in percentage
    """
    metrics = {}

    # 1. Accuracy
    metrics['ACC'] = accuracy_score(y_true, y_pred) * 100

    # 2. AUC
    try:
        if num_classes == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1]) * 100
        else:
            metrics['AUC'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            ) * 100
    except Exception as e:
        print(f"  Warning: AUC calculation failed: {e}")
        metrics['AUC'] = 0.0

    # 3. F1 Score
    if num_classes == 2:
        metrics['F1'] = f1_score(y_true, y_pred, average='binary') * 100
    else:
        metrics['F1'] = f1_score(y_true, y_pred, average='macro') * 100

    # 4. Average Precision (AP)
    try:
        if num_classes == 2:
            metrics['AP'] = average_precision_score(y_true, y_prob[:, 1]) * 100
        else:
            y_true_onehot = np.eye(num_classes)[y_true]
            metrics['AP'] = average_precision_score(
                y_true_onehot, y_prob, average='macro'
            ) * 100
    except Exception as e:
        print(f"  Warning: AP calculation failed: {e}")
        metrics['AP'] = 0.0

    return metrics


def evaluate_model_3d(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int = 2,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model

    Returns:
        metrics: {'ACC', 'AUC', 'F1', 'AP'} percentage
    """
    # Get predictions
    y_true, y_pred, y_prob = get_predictions(model, test_loader)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)

    if verbose:
        print(f"\n  Test Results:")
        print(f"    ACC: {metrics['ACC']:.2f}%")
        print(f"    AUC: {metrics['AUC']:.2f}%")
        print(f"    F1:  {metrics['F1']:.2f}%")
        print(f"    AP:  {metrics['AP']:.2f}%")

        cm = confusion_matrix(y_true, y_pred)
        print(f"    Confusion Matrix:")
        print(f"      {cm}")

    return metrics


def format_metrics(metrics: Dict[str, float], decimal: int = 2) -> str:
    """Format metrics as string (same as 2D)"""
    return (f"ACC: {metrics['ACC']:.{decimal}f}% | "
            f"AUC: {metrics['AUC']:.{decimal}f}% | "
            f"F1: {metrics['F1']:.{decimal}f}% | "
            f"AP: {metrics['AP']:.{decimal}f}%")