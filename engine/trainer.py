# engine/trainer.py
"""
Ark+-style cyclic trainer.

Training loop:
    1. Each round iterates through all datasets sequentially.
    2. Teacher-Student framework with consistency loss.
    3. Expert-aware EMA (V2) or modality-aware EMA (V1).

Key design:
    - Teacher receives NO data augmentation.
    - V2 mode: only the active expert's adapter is updated via EMA.
    - Consistency loss aligns student features toward teacher features.
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import DataLoader

from config.datasets import get_dataset_config, get_expert_id
from .evaluator import evaluate_all_datasets, print_evaluation_results


class Trainer:
    """
    Cyclic trainer for MOSAIC.

    Args:
        student / teacher: Student and EMA-teacher models.
        dataloaders:       {dataset_name: {train/val/test: DataLoader}}.
        dataset_list:      Ordered list of dataset names (training order).
        optimizer / scheduler: PyTorch optimizer and LR scheduler.
        device:            torch.device.
        args:              Namespace with training hyper-parameters.
        logger:            Optional ExperimentLogger instance.
    """

    def __init__(self, student, teacher, dataloaders, dataset_list,
                 optimizer, scheduler=None, device=None, args=None, logger=None):
        self.student = student
        self.teacher = teacher
        self.dataloaders = dataloaders
        self.dataset_list = dataset_list
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.logger = logger

        self.student.to(self.device)
        self.teacher.to(self.device)

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        self.current_round = 0
        self.global_step = 0
        self.best_acc = 0.0
        self.best_round = 0
        self.history: Dict = {"train_loss": [], "val_results": [], "test_results": []}

    # ------------------------------------------------------------------

    def train_one_epoch(self, dataset_name, task_id, epoch):
        self.student.train()
        self.teacher.eval()

        config = get_dataset_config(dataset_name)
        expert_id = get_expert_id(dataset_name)
        adapter_mode = getattr(self.student, "adapter_mode", "v1")
        loss_fn = self.bce_loss if config.task_type == "multi-label" else self.ce_loss
        loader = self.dataloaders[dataset_name]["train"]

        total_loss = total_cls = total_con = 0.0
        n = 0

        for images, labels in loader:
            # Unpack dual augmentation
            if isinstance(images, (tuple, list)) and len(images) == 2:
                imgs_s, imgs_t = images[0].to(self.device), images[1].to(self.device)
            else:
                imgs_s = imgs_t = images.to(self.device)

            labels = labels.to(self.device)
            labels = labels.float() if config.task_type == "multi-label" else labels.long().squeeze()

            # Student forward
            feat_s, logits = self.student(imgs_s, task_id=task_id, expert_id=expert_id)

            # Teacher forward (no grad)
            with torch.no_grad():
                feat_t = self.teacher(imgs_t, task_id=task_id,
                                      return_features=True, expert_id=expert_id)

            # Loss
            l_cls = loss_fn(logits, labels)
            l_con = self.mse_loss(feat_s, feat_t.detach())
            cw = getattr(self.args, "consist_weight", 0.1)
            loss = l_cls + cw * l_con

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            mg = getattr(self.args, "max_grad_norm", 1.0)
            nn.utils.clip_grad_norm_(self.student.parameters(), mg)
            self.optimizer.step()

            # EMA update
            mom = getattr(self.args, "ema_momentum", 0.999)
            mom_3d = getattr(self.args, "ema_momentum_3d", mom)
            if adapter_mode == "v2_moe":
                actual_mom = mom_3d if expert_id == "C" else mom
                self.teacher.ema_update(self.student, actual_mom, expert_id=expert_id)
            else:
                actual_mom = mom_3d if config.is_3d else mom
                self.teacher.ema_update(self.student, actual_mom, is_3d=config.is_3d)

            total_loss += loss.item()
            total_cls += l_cls.item()
            total_con += l_con.item()
            n += 1
            self.global_step += 1

        return {"loss": total_loss / n, "loss_cls": total_cls / n, "loss_consist": total_con / n}

    # ------------------------------------------------------------------

    def train_one_round(self, round_idx):
        self.current_round = round_idx
        print(f"\n{'=' * 60}\n Round {round_idx + 1}\n{'=' * 60}")
        losses = {}
        for tid, name in enumerate(self.dataset_list):
            r = self.train_one_epoch(name, tid, round_idx)
            losses[name] = r["loss"]
            print(f"  {name}: loss={r['loss']:.4f}, cls={r['loss_cls']:.4f}, "
                  f"consist={r['loss_consist']:.4f}")
        if self.scheduler is not None:
            self.scheduler.step()
        losses["mean"] = sum(losses.values()) / len(losses)
        return losses

    # ------------------------------------------------------------------

    def train(self, num_rounds, eval_every=1, save_every=10, output_dir="./output",
              early_stopping_patience=5, early_stopping_min_delta=0.001):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'#' * 60}\n Starting Training — {num_rounds} rounds, "
              f"{len(self.dataset_list)} datasets\n{'#' * 60}")

        start = time.time()
        no_improve = 0

        for r in range(num_rounds):
            rl = self.train_one_round(r)
            self.history["train_loss"].append(rl)

            if (r + 1) % eval_every == 0:
                val_s = evaluate_all_datasets(self.student, self.dataloaders,
                                              self.dataset_list, self.device, "val")
                val_t = evaluate_all_datasets(self.teacher, self.dataloaders,
                                              self.dataset_list, self.device, "val")
                print_evaluation_results(val_s, self.dataset_list,
                                         f"Student Val (Round {r+1})")
                print_evaluation_results(val_t, self.dataset_list,
                                         f"Teacher Val (Round {r+1})")

                if self.logger:
                    self.logger.log_round(r, val_s, val_t)
                self.history["val_results"].append({"round": r+1, "student": val_s, "teacher": val_t})

                cur = val_s.get("mean_acc", 0)
                if cur > self.best_acc:
                    self.best_acc = cur
                    self.best_round = r
                    self.save_checkpoint(os.path.join(output_dir, "best_model.pth"), r)
                    print(f"  ★ New best! ACC={cur*100:.2f}%")
                    no_improve = 0
                elif cur > self.best_acc - early_stopping_min_delta:
                    no_improve = 0
                else:
                    no_improve += 1
                    print(f"  No improvement ({no_improve}/{early_stopping_patience})")

                if no_improve >= early_stopping_patience:
                    print(f"\n  Early stopping at round {r+1}")
                    break

            if (r + 1) % save_every == 0:
                self.save_checkpoint(os.path.join(output_dir, f"ckpt_round{r+1}.pth"), r)

        # Final test with best model
        best_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            self.student.load_state_dict(ckpt["student_state_dict"])
            self.teacher.model.load_state_dict(ckpt["teacher_state_dict"])

        test_s = evaluate_all_datasets(self.student, self.dataloaders,
                                       self.dataset_list, self.device, "test")
        test_t = evaluate_all_datasets(self.teacher, self.dataloaders,
                                       self.dataset_list, self.device, "test")
        print_evaluation_results(test_s, self.dataset_list, "Student Test")
        print_evaluation_results(test_t, self.dataset_list, "Teacher Test")
        self.history["test_results"] = {"student": test_s, "teacher": test_t}

        if self.logger:
            self.logger.log_test_results(test_s, test_t, self.best_round, self.best_acc)
            self.logger.finalize()

        print(f"\nDone in {(time.time()-start)/3600:.2f}h — best ACC={self.best_acc*100:.2f}%")
        self._save_history(os.path.join(output_dir, "history.json"))
        self.save_checkpoint(os.path.join(output_dir, "final_model.pth"), num_rounds - 1)
        return self.history

    # ------------------------------------------------------------------

    def save_checkpoint(self, path, round_idx):
        ckpt = {
            "round": round_idx,
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
            "global_step": self.global_step,
        }
        if self.scheduler:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["student_state_dict"])
        self.teacher.model.load_state_dict(ckpt["teacher_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_acc = ckpt.get("best_acc", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.current_round = ckpt.get("round", 0)
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"Resumed from round {self.current_round + 1}")

    def _save_history(self, path):
        def _convert(o):
            if isinstance(o, (np.floating,)):   return float(o)
            if isinstance(o, (np.integer,)):    return int(o)
            if isinstance(o, np.ndarray):       return o.tolist()
            if isinstance(o, dict):             return {k: _convert(v) for k, v in o.items()}
            if isinstance(o, list):             return [_convert(v) for v in o]
            return o
        with open(path, "w") as f:
            json.dump(_convert(self.history), f, indent=2)


# ======================================================================
# Optimizer / Scheduler factories
# ======================================================================

def create_optimizer(model, lr=1e-4, weight_decay=0.01,
                     separate_adapter_lr=False, adapter_lr_multiplier=10.0):
    if separate_adapter_lr:
        adapt = [p for n, p in model.named_parameters() if "adapter" in n]
        other = [p for n, p in model.named_parameters() if "adapter" not in n]
        groups = [{"params": other, "lr": lr},
                  {"params": adapt, "lr": lr * adapter_lr_multiplier}]
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, num_rounds, warmup_rounds=5, min_lr=1e-6):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_rounds)
    cosine = CosineAnnealingLR(optimizer, T_max=num_rounds - warmup_rounds,
                               eta_min=min_lr)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_rounds])