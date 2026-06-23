#!/usr/bin/env python3
# external_eval/run_3d.py
"""
3D External Validation main script

Run all models × 3 seeds:
  1. ImageNet ViT 3D
  2. Med3D ResNet-18
  3. Med3D ResNet-34
  4. Med3D ResNet-50
  5. Ours (Expert C)

Dataset: MosMedData COVID-19 CT
Metrics: ACC, AUC, F1, AP (same as 2D)

Save: JSON + CSV + LaTeX
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import torch

from configs_3d import (
    MODELS_3D, DATASET_3D, TRAIN_CONFIG,
    RESULTS_ROOT, RANDOM_SEEDS, DEVICE,
    get_available_models, print_config,
)
from dataset_3d import get_dataloader_3d
from models_3d import create_model_3d
from train_3d import train_linear_probe, set_seed
from evaluate_3d import evaluate_model_3d, format_metrics


def run_single(model_name, seed, epochs=None, verbose=True, finetune=False):
    """
    Run single experiment (one model × one seed)

    Args:
        finetune: True = full fine-tuning (Unfreeze backbone), False = linear probing
    """
    mode_str = 'finetune' if finetune else 'linear_probe'
    result = {
        'model': model_name,
        'seed': seed,
        'mode': mode_str,
        'status': 'failed',
        'train_time': 0,
        'best_val_acc': 0,
        'test_metrics': {},
    }

    try:
        set_seed(seed)
        start = time.time()

        # 1. Data
        train_loader = get_dataloader_3d('train')
        val_loader = get_dataloader_3d('val')
        test_loader = get_dataloader_3d('test')

        # 2. Model
        num_classes = DATASET_3D['num_classes']
        cfg = MODELS_3D[model_name]

        # Ours: fine-tuning (3-tier LR: backbone small, adapter mid, head large)
        # Others: fine-tuning (2-tier LR: backbone small, head large)
        is_ours = cfg['arch'] == 'unified_model_moe'
        model = create_model_3d(model_name, num_classes=num_classes,
                                freeze_backbone=not finetune)

        # 3. Train
        # Ours: backbone=1e-5, adapter=1e-4, head=1e-3 (3-tier LR)
        # ImageNet ViT: backbone=1e-4, head=1e-3
        # ResNet: backbone=1e-5, head=1e-3
        if finetune and is_ours:
            bk_lr = 1e-5       # backbone (attention+FFN) conservative
            adapter_lr = 1e-4  # adapter moderate
        elif finetune and cfg['arch'] in ('unified_model',):
            bk_lr = 1e-4   # ImageNet ViT backbone
            adapter_lr = None
        elif finetune:
            bk_lr = 1e-5   # ResNet backbone
            adapter_lr = None
        else:
            bk_lr = None
            adapter_lr = None

        model, history = train_linear_probe(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            epochs=epochs,
            verbose=verbose,
            finetune=finetune,
            backbone_lr=bk_lr,
            adapter_lr=adapter_lr,
        )

        elapsed = time.time() - start

        # 4. test
        metrics = evaluate_model_3d(
            model=model,
            test_loader=test_loader,
            num_classes=num_classes,
            verbose=verbose,
        )

        result['status'] = 'success'
        result['train_time'] = elapsed
        result['best_val_acc'] = max(history['val_acc'])
        result['test_metrics'] = metrics

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)

    return result


def run_all(model_names=None, seeds=None, epochs=None, verbose=True, finetune=False):
    """Run all experiments"""
    if model_names is None:
        model_names = get_available_models()
    if seeds is None:
        seeds = RANDOM_SEEDS

    mode_str = 'Fine-tuning' if finetune else 'Linear Probe'
    tag = 'ft' if finetune else 'lp'

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    print_config()

    print(f"\n{'='*60}")
    print(f"Starting 3D External Validation ({mode_str})")
    print(f"Models: {model_names}")
    print(f"Seeds:  {seeds}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    all_results = []
    total_start = time.time()

    for model_name in model_names:
        cfg = MODELS_3D[model_name]
        print(f"\n{'#'*60}")
        print(f"# Model: {cfg['display_name']}")
        print(f"# Pretrain: {cfg['pretrain_data']}")
        print(f"{'#'*60}")

        for seed in seeds:
            print(f"\n  --- Seed: {seed} ---")
            result = run_single(model_name, seed, epochs=epochs,
                                verbose=verbose, finetune=finetune)
            all_results.append(result)

            if result['status'] == 'success':
                print(f"  => {format_metrics(result['test_metrics'])} "
                      f"({result['train_time']:.1f}s)")
            else:
                print(f"  => FAILED: {result.get('error', 'unknown')}")

    total_time = time.time() - total_start

    # ============================================================
    # Summary: mean ± std
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Results Summary - {mode_str} (MosMedData COVID-19 CT)")
    print(f"{'='*60}")

    summary = {}
    for model_name in model_names:
        cfg = MODELS_3D[model_name]
        model_results = [r for r in all_results
                         if r['model'] == model_name and r['status'] == 'success']

        if not model_results:
            print(f"\n{cfg['display_name']}: ALL FAILED")
            continue

        accs = [r['test_metrics']['ACC'] for r in model_results]
        aucs = [r['test_metrics']['AUC'] for r in model_results]
        f1s  = [r['test_metrics']['F1']  for r in model_results]
        aps  = [r['test_metrics']['AP']  for r in model_results]

        summary[model_name] = {
            'display_name': cfg['display_name'],
            'pretrain_data': cfg['pretrain_data'],
            'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
            'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
            'f1_mean':  np.mean(f1s),  'f1_std':  np.std(f1s),
            'ap_mean':  np.mean(aps),  'ap_std':  np.std(aps),
            'n_seeds': len(model_results),
        }

        print(f"\n{cfg['display_name']} ({cfg['pretrain_data']})")
        print(f"  ACC: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
        print(f"  AUC: {np.mean(aucs):.2f} ± {np.std(aucs):.2f}")
        print(f"  F1:  {np.mean(f1s):.2f} ± {np.std(f1s):.2f}")
        print(f"  AP:  {np.mean(aps):.2f} ± {np.std(aps):.2f}")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ============================================================
    # Save results
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- JSON ---
    json_path = os.path.join(RESULTS_ROOT, f'results_3d_{tag}_{timestamp}.json')
    save_data = {
        'timestamp': timestamp,
        'mode': mode_str,
        'dataset': DATASET_3D['name'],
        'train_config': TRAIN_CONFIG,
        'seeds': seeds,
        'total_time': total_time,
        'summary': summary,
        'detailed_results': all_results,
    }
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nDetailed results: {json_path}")

    # --- CSV ---
    csv_path = os.path.join(RESULTS_ROOT, f'results_3d_{tag}_{timestamp}.csv')
    with open(csv_path, 'w') as f:
        f.write("Model,Pretrain,ACC_mean,ACC_std,AUC_mean,AUC_std,"
                "F1_mean,F1_std,AP_mean,AP_std\n")
        for model_name, s in summary.items():
            f.write(f"{s['display_name']},{s['pretrain_data']},"
                    f"{s['acc_mean']:.2f},{s['acc_std']:.2f},"
                    f"{s['auc_mean']:.2f},{s['auc_std']:.2f},"
                    f"{s['f1_mean']:.2f},{s['f1_std']:.2f},"
                    f"{s['ap_mean']:.2f},{s['ap_std']:.2f}\n")
    print(f"Summary CSV:      {csv_path}")

    # --- LaTeX ---
    latex_path = os.path.join(RESULTS_ROOT, f'table_3d_{tag}_{timestamp}.tex')
    with open(latex_path, 'w') as f:
        f.write(f"% 3D External Validation ({mode_str}) - MosMedData COVID-19 CT\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{3D External Validation ({mode_str}) on MosMedData COVID-19 CT}}\n")
        f.write(f"\\label{{tab:3d_external_{tag}}}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & ACC (\\%) & AUC (\\%) & F1 (\\%) & AP (\\%) \\\\\n")
        f.write("\\midrule\n")
        for model_name, s in summary.items():
            is_ours = 'ours' in model_name
            bf = "\\textbf" if is_ours else ""
            def _fmt(mean, std):
                txt = f"{mean:.2f}$\\pm${std:.2f}"
                return f"{bf}{{{txt}}}" if is_ours else txt
            disp = "\\textbf{" + s["display_name"] + "}" if is_ours else s["display_name"]
            f.write(f"{disp} & "
                    f"{_fmt(s['acc_mean'], s['acc_std'])} & "
                    f"{_fmt(s['auc_mean'], s['auc_std'])} & "
                    f"{_fmt(s['f1_mean'], s['f1_std'])} & "
                    f"{_fmt(s['ap_mean'], s['ap_std'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table:      {latex_path}")

    return all_results, summary


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D External Validation')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to evaluate (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: from config)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 1 seed, 5 epochs, med3d_resnet50 only')
    parser.add_argument('--med3d-only', action='store_true',
                        help='Only run Med3D models (skip ViT models)')
    parser.add_argument('--finetune', action='store_true',
                        help='Full fine-tuning (unfreeze backbone). Default: linear probe')
    args = parser.parse_args()

    if args.quick:
        models = ['med3d_resnet50']
        seeds = [42]
        epochs = 5
        print("=== QUICK TEST MODE ===")
    elif args.med3d_only:
        models = ['med3d_resnet18', 'med3d_resnet34', 'med3d_resnet50']
        seeds = args.seeds
        epochs = args.epochs
    else:
        models = args.models
        seeds = args.seeds
        epochs = args.epochs

    run_all(model_names=models, seeds=seeds, epochs=epochs, finetune=args.finetune)