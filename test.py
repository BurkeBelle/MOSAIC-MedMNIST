#!/usr/bin/env python3
# test.py
"""
Evaluation script — load a saved checkpoint and evaluate on test/val set.

Usage:
    python test.py --checkpoint ./output/best_model.pth --data_root ./data_224

    # Evaluate only 2D datasets
    python test.py --checkpoint ./output/best_model.pth --only_2d

    # Evaluate on validation set
    python test.py --checkpoint ./output/best_model.pth --split val
"""

import os
import argparse
import torch
import numpy as np

from config.datasets import (
    ALL_DATASETS, DATASETS_2D, DATASETS_3D,
    get_num_classes_list, get_dataset_config
)
from dataloader.medmnist_loader import create_all_dataloaders
from model.unified_model import create_model_and_teacher
from engine.evaluator import evaluate_all_datasets, print_evaluation_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MOSAIC model")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, default="./data_224",
                        help="Root directory for MedMNIST data")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset_list", nargs="+", default=None,
                        help="Specific datasets to evaluate (default: all 18)")
    parser.add_argument("--only_2d", action="store_true",
                        help="Evaluate only 2D datasets")
    parser.add_argument("--only_3d", action="store_true",
                        help="Evaluate only 3D datasets")
    parser.add_argument("--use_adapter", action="store_true", default=True,
                        help="Model uses adapter (default: True)")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Model does not use adapter")
    parser.add_argument("--adapter_bottleneck", type=int, default=64,
                        help="Adapter bottleneck dimension")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Evaluation split (val or test)")

    args = parser.parse_args()
    if args.no_adapter:
        args.use_adapter = False
    return args


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine dataset list
    if args.dataset_list is not None:
        dataset_list = args.dataset_list
    elif args.only_2d:
        dataset_list = DATASETS_2D
    elif args.only_3d:
        dataset_list = DATASETS_3D
    else:
        dataset_list = ALL_DATASETS

    print(f"\nDatasets to evaluate: {len(dataset_list)}")

    # 1. Load data
    print(f"\n{'=' * 60}\n Loading Data\n{'=' * 60}")
    dataloaders = create_all_dataloaders(
        dataset_list=dataset_list,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dual_transform=False,
    )

    # 2. Create model
    print(f"\n{'=' * 60}\n Creating Model\n{'=' * 60}")
    num_classes_list = get_num_classes_list(dataset_list)
    student, teacher = create_model_and_teacher(
        num_classes_list=num_classes_list,
        use_adapter=args.use_adapter,
        adapter_bottleneck=args.adapter_bottleneck,
    )

    # 3. Load checkpoint
    print(f"\n{'=' * 60}\n Loading Checkpoint\n{'=' * 60}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"  Path: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "round" in checkpoint:
        print(f"  Round: {checkpoint['round'] + 1}")
    if "best_acc" in checkpoint:
        print(f"  Best ACC: {checkpoint['best_acc']*100:.2f}%")
    if "global_step" in checkpoint:
        print(f"  Global Step: {checkpoint['global_step']}")

    student.load_state_dict(checkpoint["student_state_dict"])
    print("  Student model loaded")

    if "teacher_state_dict" in checkpoint:
        teacher.model.load_state_dict(checkpoint["teacher_state_dict"])
        print("  Teacher model loaded")

    student.to(device)
    teacher.to(device)

    # 4. Evaluate
    print(f"\n{'=' * 60}\n Evaluating on {args.split} set\n{'=' * 60}")

    student_results = evaluate_all_datasets(
        model=student, dataloaders=dataloaders,
        dataset_list=dataset_list, device=device, split=args.split,
    )
    print_evaluation_results(student_results, dataset_list,
                             title=f"Student {args.split.capitalize()} Results")

    teacher_results = evaluate_all_datasets(
        model=teacher, dataloaders=dataloaders,
        dataset_list=dataset_list, device=device, split=args.split,
    )
    print_evaluation_results(teacher_results, dataset_list,
                             title=f"Teacher {args.split.capitalize()} Results")

    # 5. Summary
    print(f"\n{'=' * 60}\n Summary\n{'=' * 60}")

    print(f"\n  Student:")
    print(f"    Mean ACC: {student_results['mean_acc']*100:.2f}%")
    print(f"    Mean AUC: {student_results['mean_auc']:.4f}")
    if "mean_acc_2d" in student_results:
        print(f"    2D ACC:   {student_results['mean_acc_2d']*100:.2f}%")
    if "mean_acc_3d" in student_results:
        print(f"    3D ACC:   {student_results['mean_acc_3d']*100:.2f}%")

    print(f"\n  Teacher:")
    print(f"    Mean ACC: {teacher_results['mean_acc']*100:.2f}%")
    print(f"    Mean AUC: {teacher_results['mean_auc']:.4f}")

    gap = student_results["mean_acc"] - teacher_results["mean_acc"]
    print(f"\n  Gap (S-T): {gap*100:+.2f}%")

    print(f"\n{'=' * 60}\n Evaluation Complete\n{'=' * 60}")

    return student_results, teacher_results


if __name__ == "__main__":
    main()