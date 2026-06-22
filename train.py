#!/usr/bin/env python3
# train.py
"""
MOSAIC training entry point.

Usage:
    # MoE three-expert (default, reproduces paper results)
    python train.py --freeze_backbone --pretrained ./vit_base_patch16_224.npz

    # V1 dual-channel adapter
    python train.py --adapter_mode v1 --freeze_backbone --pretrained ./vit_base_patch16_224.npz

    # 2D-3D interleaved order
    python train.py --use_interleaved --freeze_backbone --pretrained ./vit_base_patch16_224.npz
"""

import os
import json
import argparse
import random
import numpy as np
import torch

from config.datasets import (
    ALL_DATASETS, INTERLEAVED_DATASETS, DATASETS_2D, DATASETS_3D,
    get_num_classes_list,
)
from dataloader.medmnist_loader import create_all_dataloaders
from model.unified_model import create_model_and_teacher
from engine.trainer import Trainer, create_optimizer, create_scheduler
from utils.logger import ExperimentLogger


def parse_args():
    p = argparse.ArgumentParser(description="MOSAIC Training")

    # Data
    p.add_argument("--data_root", type=str, default="./data_224")
    p.add_argument("--dataset_list", nargs="+", default=None)
    p.add_argument("--use_interleaved", action="store_true")
    p.add_argument("--only_2d", action="store_true")
    p.add_argument("--only_3d", action="store_true")

    # Model
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--adapter_mode", type=str, default="v2_moe", choices=["v1", "v2_moe"])
    p.add_argument("--adapter_bottleneck", type=int, default=64)
    p.add_argument("--adapter_bottleneck_a", type=int, default=64)
    p.add_argument("--adapter_bottleneck_b", type=int, default=96)
    p.add_argument("--adapter_bottleneck_c", type=int, default=192)
    p.add_argument("--adapter_scalar", type=float, default=0.1)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--no_adapter", action="store_true")

    # Training
    p.add_argument("--num_rounds", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_rounds", type=int, default=5)
    p.add_argument("--consist_weight", type=float, default=0.1)
    p.add_argument("--ema_momentum", type=float, default=0.9)
    p.add_argument("--ema_momentum_3d", type=float, default=0.95)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--early_stopping_min_delta", type=float, default=0.001)

    # Misc
    p.add_argument("--exp_name", type=str, default="mosaic")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", type=str, default=None)

    args = p.parse_args()
    if args.no_adapter:
        args.use_adapter = False
    else:
        args.use_adapter = True
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine dataset list
    if args.dataset_list:
        dataset_list = args.dataset_list
    elif args.only_2d:
        dataset_list = DATASETS_2D
    elif args.only_3d:
        dataset_list = DATASETS_3D
    elif args.use_interleaved:
        dataset_list = INTERLEAVED_DATASETS
    else:
        dataset_list = ALL_DATASETS

    print(f"Device: {device}")
    print(f"Datasets ({len(dataset_list)}): {dataset_list}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    dataloaders = create_all_dataloaders(
        dataset_list, args.data_root, args.batch_size, args.num_workers, dual_transform=True
    )

    # Model
    num_classes_list = get_num_classes_list(dataset_list)
    student, teacher = create_model_and_teacher(
        num_classes_list=num_classes_list,
        use_adapter=args.use_adapter,
        adapter_mode=args.adapter_mode,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_bottleneck_a=args.adapter_bottleneck_a,
        adapter_bottleneck_b=args.adapter_bottleneck_b,
        adapter_bottleneck_c=args.adapter_bottleneck_c,
        adapter_scalar=args.adapter_scalar,
        pretrained_path=args.pretrained,
    )
    if args.freeze_backbone:
        student.freeze_backbone()

    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable "
          f"({trainable/total*100:.1f}%)")

    # Optimizer & scheduler
    optimizer = create_optimizer(student, args.lr, args.weight_decay)
    scheduler = create_scheduler(optimizer, args.num_rounds, args.warmup_rounds)

    # Logger
    logger = ExperimentLogger(args.exp_name, dataset_list, args.output_dir, vars(args))

    # Trainer
    trainer = Trainer(student, teacher, dataloaders, dataset_list,
                      optimizer, scheduler, device, args, logger)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        num_rounds=args.num_rounds,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


if __name__ == "__main__":
    main()