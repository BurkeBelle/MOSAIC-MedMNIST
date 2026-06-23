"""
Full Fine-tuning External Validation

Unlike linear probe, the entire backbone is unfrozen for training
Requires smaller LR and longer training time

Note: Ours model only trains Adapter + Head (backbone frozen)
"""

import os
import json
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch

from configs import DATASETS, RESULTS_ROOT, DEVICE, RANDOM_SEEDS
from dataset import get_dataloader
from models import create_model, get_available_models
from train import set_seed
from evaluate import evaluate_model


# Full fine-tuning config (differs from linear probe)
FINETUNE_CONFIG = {
    'epochs': 100,           # more epochs
    'batch_size': 32,
    'lr': 1e-4,              # smaller LR (important!)
    'lr_adapter': 1e-3,      # Adapter LR (for Ours model)
    'weight_decay': 1e-4,
    'patience': 15,          # longer patience
    'scheduler': 'cosine',
}


def train_finetune(
    model,
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int = FINETUNE_CONFIG['epochs'],
    lr: float = FINETUNE_CONFIG['lr'],
    weight_decay: float = FINETUNE_CONFIG['weight_decay'],
    patience: int = FINETUNE_CONFIG['patience'],
    device: str = DEVICE,
    verbose: bool = True,
    is_adapter_tuning: bool = False,  # whether adapter tuning mode
):
    """
    Full fine-tuning training
    
    Args:
        is_adapter_tuning: True = train adapter + head only (Ours model)
    """
    import torch.nn as nn
    import torch.optim as optim
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if is_adapter_tuning:
        # Ours: train only adapter+head with larger LR
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=FINETUNE_CONFIG['lr_adapter'],  # 1e-3
            weight_decay=weight_decay
        )
        if verbose:
            print(f"  [Adapter Tuning] lr={FINETUNE_CONFIG['lr_adapter']}, "
                  f"trainable params: {sum(p.numel() for p in trainable_params):,}")
    else:
        # Baselines: train all parameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,  # 1e-4
            weight_decay=weight_decay
        )
        if verbose:
            print(f"  [Full Fine-tuning] lr={lr}, "
                  f"total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Cosine LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early stopping
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    for epoch in range(epochs):
        # ===== Training =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Check improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = '*'
        else:
            patience_counter += 1
            marker = ''
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}% {marker}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def run_single_experiment_finetune(
    dataset_name: str,
    model_name: str,
    seed: int,
    epochs: int = FINETUNE_CONFIG['epochs'],
    batch_size: int = FINETUNE_CONFIG['batch_size'],
    verbose: bool = True
) -> dict:
    """
    Run single fine-tuning experiment
    """
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'status': 'failed',
        'train_time': 0,
        'best_val_acc': 0,
        'test_metrics': {}
    }
    
    try:
        # Set seed
        set_seed(seed)
        
        start_time = time.time()
        
        # 1. Load data
        train_loader = get_dataloader(dataset_name, 'train', batch_size=batch_size)
        val_loader = get_dataloader(dataset_name, 'val', batch_size=batch_size)
        test_loader = get_dataloader(dataset_name, 'test', batch_size=batch_size)
        
        # 2. Create model
        num_classes = DATASETS[dataset_name]['num_classes']
        
        # ========== Key modification ==========
        if model_name == 'ours':
            # Ours: Adapter Tuning (backbone frozen, train adapter+head)
            model = create_model(model_name, num_classes=num_classes, freeze_backbone=True)
            is_adapter_tuning = True
            if verbose:
                print(f"  [Ours] Adapter Tuning mode (backbone frozen)")
        else:
            # Baselines: Full Fine-tuning (train all params)
            model = create_model(model_name, num_classes=num_classes, freeze_backbone=False)
            is_adapter_tuning = False
            if verbose:
                print(f"  [{model_name}] Full Fine-tuning mode")
        # ==============================
        
        # 2.1 Set correct expert for ours model
        if model_name == 'ours':
            expert_id = DATASETS[dataset_name].get('expert', 'A')
            model.set_expert(expert_id)
            if verbose:
                print(f"  [Ours] Using Expert {expert_id}")
        
        # 3. Train
        model, history = train_finetune(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            epochs=epochs,
            verbose=verbose,
            is_adapter_tuning=is_adapter_tuning,
        )
        
        train_time = time.time() - start_time
        
        # 4. Evaluate on test set
        test_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            num_classes=num_classes,
            verbose=verbose
        )
        
        # 5. Save results
        result['status'] = 'success'
        result['train_time'] = train_time
        result['best_val_acc'] = max(history['val_acc'])
        result['test_metrics'] = test_metrics
        result['epochs_trained'] = len(history['train_loss'])
        result['tuning_mode'] = 'adapter' if is_adapter_tuning else 'full'
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        if verbose:
            print(f"\n[Error] {e}")
            import traceback
            traceback.print_exc()
    
    return result


def run_all_finetune_experiments(
    datasets: list = None,
    models: list = None,
    seeds: list = RANDOM_SEEDS,
    epochs: int = FINETUNE_CONFIG['epochs'],
    batch_size: int = FINETUNE_CONFIG['batch_size'],
    save_dir: str = RESULTS_ROOT,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run all fine-tuning experiments (multi-seed)
    """
    if datasets is None:
        datasets = list(DATASETS.keys())
    if models is None:
        models = get_available_models()
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    total_exp = len(datasets) * len(models) * len(seeds)
    
    print("=" * 70)
    print("External Validation - Fine-tuning (Multi-Seed)")
    print("=" * 70)
    print(f"  Datasets: {datasets}")
    print(f"  Models: {models}")
    print(f"  Seeds: {seeds}")
    print(f"  Total experiments: {total_exp}")
    print(f"  Epochs: {epochs}")
    print(f"  Baselines: Full Fine-tuning (lr={FINETUNE_CONFIG['lr']})")
    print(f"  Ours: Adapter Tuning (lr={FINETUNE_CONFIG['lr_adapter']})")
    print(f"  Device: {DEVICE}")
    print("=" * 70)
    
    all_results = []
    total_start_time = time.time()
    exp_idx = 0
    
    for dataset_name in datasets:
        for model_name in models:
            for seed in seeds:
                exp_idx += 1
                
                print(f"\n{'#'*70}")
                print(f"# [{exp_idx}/{total_exp}] {dataset_name} + {model_name} (seed={seed})")
                print(f"{'#'*70}")
                
                result = run_single_experiment_finetune(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose
                )
                
                all_results.append(result)
                
                # Save intermediate results
                _save_intermediate_finetune(all_results, save_dir, timestamp)
                
                # Print progress
                elapsed = time.time() - total_start_time
                avg_time = elapsed / exp_idx
                remaining = avg_time * (total_exp - exp_idx)
                print(f"  [Progress] {exp_idx}/{total_exp} done, "
                      f"elapsed: {elapsed/60:.1f}min, remaining: {remaining/60:.1f}min")
    
    total_time = time.time() - total_start_time
    
    # Save full results
    results_df = save_finetune_results(all_results, save_dir, timestamp)
    
    # Print summary
    print_finetune_summary(results_df, seeds, total_time)
    
    return results_df


def _save_intermediate_finetune(results: list, save_dir: str, timestamp: str):
    """Save intermediate results"""
    json_path = os.path.join(save_dir, f'finetune_{timestamp}_intermediate.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def save_finetune_results(results: list, save_dir: str, timestamp: str) -> pd.DataFrame:
    """Save fine-tuning results"""
    # 1. Save full JSON
    json_path = os.path.join(save_dir, f'finetune_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] JSON: {json_path}")
    
    # 2. Convert to DataFrame
    rows = []
    for r in results:
        if r['status'] == 'success':
            row = {
                'Dataset': r['dataset'],
                'Model': r['model'],
                'Seed': r['seed'],
                'ACC': r['test_metrics'].get('ACC', 0),
                'AUC': r['test_metrics'].get('AUC', 0),
                'F1': r['test_metrics'].get('F1', 0),
                'AP': r['test_metrics'].get('AP', 0),
                'Epochs': r.get('epochs_trained', 0),
                'Time': r['train_time'],
                'Mode': r.get('tuning_mode', 'full'),
            }
        else:
            row = {
                'Dataset': r['dataset'],
                'Model': r['model'],
                'Seed': r['seed'],
                'ACC': 0, 'AUC': 0, 'F1': 0, 'AP': 0,
                'Epochs': 0, 'Time': 0, 'Mode': 'failed',
            }
        rows.append(row)
    
    results_df = pd.DataFrame(rows)
    
    # 3. Save raw CSV
    csv_path = os.path.join(save_dir, f'finetune_{timestamp}_raw.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"[Saved] Raw CSV: {csv_path}")
    
    # 4. Compute mean ± std
    summary_rows = []
    datasets = results_df['Dataset'].unique()
    models = results_df['Model'].unique()
    
    for dataset in datasets:
        for model in models:
            subset = results_df[(results_df['Dataset'] == dataset) & (results_df['Model'] == model)]
            if len(subset) > 0:
                summary_rows.append({
                    'Dataset': dataset,
                    'Model': model,
                    'ACC_mean': subset['ACC'].mean(),
                    'ACC_std': subset['ACC'].std(),
                    'AUC_mean': subset['AUC'].mean(),
                    'AUC_std': subset['AUC'].std(),
                    'F1_mean': subset['F1'].mean(),
                    'F1_std': subset['F1'].std(),
                    'AP_mean': subset['AP'].mean(),
                    'AP_std': subset['AP'].std(),
                })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(save_dir, f'finetune_{timestamp}_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Summary CSV: {summary_csv_path}")
    
    # 5. Remove intermediate files
    intermediate_path = os.path.join(save_dir, f'finetune_{timestamp}_intermediate.json')
    if os.path.exists(intermediate_path):
        os.remove(intermediate_path)
    
    return summary_df


def print_finetune_summary(summary_df: pd.DataFrame, seeds: list, total_time: float):
    """Print fine-tuning summary"""
    print("\n" + "=" * 90)
    print(f"FINE-TUNING SUMMARY (mean ± std over {len(seeds)} seeds)")
    print("=" * 90)
    print("Note: Baselines use Full Fine-tuning, Ours uses Adapter Tuning")
    
    datasets = ['bus', 'fundus', 'glaucoma', 'mammo_calc', 'mammo_mass']
    models = ['imagenet_vit', 'ours', 'radimagenet_resnet50', 'radimagenet_densenet121', 'radimagenet_inceptionv3']
    
    # ACC table
    print("\n[ACC (%) - mean ± std]")
    print(f"{'Model':<28} | {'bus':>12} | {'fundus':>12} | {'glaucoma':>12} | {'mammo_calc':>12} | {'mammo_mass':>12} | {'Average':>8}")
    print("-" * 115)
    
    for model in models:
        row_data = []
        means = []
        for dataset in datasets:
            subset = summary_df[(summary_df['Dataset'] == dataset) & (summary_df['Model'] == model)]
            if len(subset) > 0:
                mean = subset['ACC_mean'].values[0]
                std = subset['ACC_std'].values[0]
                row_data.append(f"{mean:.1f}±{std:.1f}")
                means.append(mean)
            else:
                row_data.append("N/A")
        avg_mean = np.mean(means) if means else 0
        print(f"{model:<28} | {row_data[0]:>12} | {row_data[1]:>12} | {row_data[2]:>12} | {row_data[3]:>12} | {row_data[4]:>12} | {avg_mean:>8.2f}")
    
    # AUC table
    print("\n[AUC (%) - mean ± std]")
    print(f"{'Model':<28} | {'bus':>12} | {'fundus':>12} | {'glaucoma':>12} | {'mammo_calc':>12} | {'mammo_mass':>12} | {'Average':>8}")
    print("-" * 115)
    
    for model in models:
        row_data = []
        means = []
        for dataset in datasets:
            subset = summary_df[(summary_df['Dataset'] == dataset) & (summary_df['Model'] == model)]
            if len(subset) > 0:
                mean = subset['AUC_mean'].values[0]
                std = subset['AUC_std'].values[0]
                row_data.append(f"{mean:.1f}±{std:.1f}")
                means.append(mean)
            else:
                row_data.append("N/A")
        avg_mean = np.mean(means) if means else 0
        print(f"{model:<28} | {row_data[0]:>12} | {row_data[1]:>12} | {row_data[2]:>12} | {row_data[3]:>12} | {row_data[4]:>12} | {avg_mean:>8.2f}")
    
    print(f"\n[Total Time] {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description='External Validation - Fine-tuning')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to evaluate (default: all)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to evaluate (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int, default=RANDOM_SEEDS,
                        help='Random seeds (default: 42, 123, 456)')
    parser.add_argument('--epochs', type=int, default=FINETUNE_CONFIG['epochs'],
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=FINETUNE_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--save_dir', type=str, default=RESULTS_ROOT,
                        help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')
    
    args = parser.parse_args()
    
    results_df = run_all_finetune_experiments(
        datasets=args.datasets,
        models=args.models,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        verbose=not args.quiet
    )
    
    return results_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Quick test
        print("=" * 70)
        print("Quick Test (1 dataset, 2 models, 1 seed, 10 epochs)")
        print("=" * 70)
        
        results_df = run_all_finetune_experiments(
            datasets=['glaucoma'],
            models=['imagenet_vit', 'ours'],
            seeds=[42],
            epochs=10,
            verbose=True
        )
        
        print("\n[Test completed]")
        print(results_df)
    else:
        main()