# engine/evaluator.py
"""
Evaluation module.

Supports multiple task types: multi-class, binary-class, multi-label,
ordinal-regression (with +/-1 tolerance).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config.datasets import get_dataset_config, get_expert_id


def compute_accuracy(preds, labels, task_type):
    if task_type == "multi-label":
        return float(((preds > 0).astype(int) == labels).mean())
    if task_type == "ordinal-regression":
        return float((np.abs(preds.argmax(1) - labels) <= 1).mean())
    return float((preds.argmax(1) == labels).mean())


def compute_auc(preds, labels, task_type, num_classes):
    try:
        if task_type == "multi-label":
            probs = torch.sigmoid(torch.tensor(preds)).numpy()
            aucs = [roc_auc_score(labels[:, i], probs[:, i])
                    for i in range(num_classes) if len(np.unique(labels[:, i])) > 1]
            return float(np.mean(aucs)) if aucs else 0.0
        probs = torch.softmax(torch.tensor(preds), dim=1).numpy()
        if task_type == "binary-class":
            return float(roc_auc_score(labels, probs[:, 1]))
        # multi-class: one-vs-rest
        ohe = np.zeros((len(labels), num_classes))
        ohe[np.arange(len(labels)), labels.astype(int)] = 1
        aucs = [roc_auc_score(ohe[:, i], probs[:, i])
                for i in range(num_classes) if len(np.unique(ohe[:, i])) > 1]
        return float(np.mean(aucs)) if aucs else 0.0
    except Exception as e:
        print(f"Warning: AUC computation failed: {e}")
        return 0.0


@torch.no_grad()
def evaluate_single_dataset(model, dataloader, dataset_name, task_id, device):
    model.eval()
    cfg = get_dataset_config(dataset_name)
    expert_id = get_expert_id(dataset_name)
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        if isinstance(images, (tuple, list)):
            images = images[0]
        images = images.to(device)
        _, logits = model(images, task_id=task_id, expert_id=expert_id)
        all_preds.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {
        "acc": compute_accuracy(preds, labels, cfg.task_type),
        "auc": compute_auc(preds, labels, cfg.task_type, cfg.num_classes),
    }


@torch.no_grad()
def evaluate_all_datasets(model, dataloaders, dataset_list, device, split="val"):
    model.eval()
    results = {}
    for tid, name in enumerate(dataset_list):
        if name in dataloaders:
            results[name] = evaluate_single_dataset(
                model, dataloaders[name][split], name, tid, device
            )

    if results:
        accs = [r["acc"] for r in results.values()]
        aucs = [r["auc"] for r in results.values()]
        results["mean_acc"] = float(np.mean(accs))
        results["mean_auc"] = float(np.mean(aucs))

        acc_2d = [r["acc"] for n, r in results.items()
                  if n in dataset_list and not get_dataset_config(n).is_3d]
        acc_3d = [r["acc"] for n, r in results.items()
                  if n in dataset_list and get_dataset_config(n).is_3d]
        if acc_2d:
            results["mean_acc_2d"] = float(np.mean(acc_2d))
        if acc_3d:
            results["mean_acc_3d"] = float(np.mean(acc_3d))
    return results


def print_evaluation_results(results, dataset_list, title="Evaluation"):
    print(f"\n{'=' * 70}\n {title}\n{'=' * 70}")
    print(f"{'Dataset':<20} {'ACC':>10} {'AUC':>10} {'Type':>6}")
    print("-" * 70)
    for name in dataset_list:
        if name in results:
            cfg = get_dataset_config(name)
            r = results[name]
            print(f"{name:<20} {r['acc']*100:>9.2f}% {r['auc']:>10.4f} "
                  f"{'3D' if cfg.is_3d else '2D':>6}")
    print("-" * 70)
    if "mean_acc" in results:
        print(f"{'Mean (All)':<20} {results['mean_acc']*100:>9.2f}% "
              f"{results['mean_auc']:>10.4f}")
    if "mean_acc_2d" in results:
        print(f"{'Mean (2D)':<20} {results['mean_acc_2d']*100:>9.2f}%")
    if "mean_acc_3d" in results:
        print(f"{'Mean (3D)':<20} {results['mean_acc_3d']*100:>9.2f}%")
    print("=" * 70 + "\n")