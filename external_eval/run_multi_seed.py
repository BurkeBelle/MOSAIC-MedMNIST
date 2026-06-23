# external_eval/run_multi_seed.py
"""
Multi-seed External Validation

Run 3-seed experiments, compute mean ± std
"""

import os
import json
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch

from configs import DATASETS, TRAIN_CONFIG, RESULTS_ROOT, DEVICE, RANDOM_SEEDS
from dataset import get_dataloader
from models import create_model, get_available_models
from train import train_linear_probe, set_seed
from evaluate import evaluate_model


def run_single_experiment_with_seed(
    dataset_name: str,
    model_name: str,
    seed: int,
    epochs: int = TRAIN_CONFIG['epochs'],
    batch_size: int = TRAIN_CONFIG['batch_size'],
    verbose: bool = True
) -> dict:
    """
    Run single experiment (specified seed)
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
        model = create_model(model_name, num_classes=num_classes, freeze_backbone=True)
        
        # 2.1 Set correct expert for ours model
        if model_name == 'ours':
            expert_id = DATASETS[dataset_name].get('expert', 'A')
            model.set_expert(expert_id)
        
        # 3. Train
        model, history = train_linear_probe(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            epochs=epochs,
            verbose=verbose
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


def run_all_experiments_multi_seed(
    datasets: list = None,
    models: list = None,
    seeds: list = RANDOM_SEEDS,
    epochs: int = TRAIN_CONFIG['epochs'],
    batch_size: int = TRAIN_CONFIG['batch_size'],
    save_dir: str = RESULTS_ROOT,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run all experiments (multi-seed)
    """
    if datasets is None:
        datasets = list(DATASETS.keys())
    if models is None:
        models = get_available_models()
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    total_exp = len(datasets) * len(models) * len(seeds)
    
    print("=" * 70)
    print("External Validation - Multi-Seed Experiments")
    print("=" * 70)
    print(f"  Datasets: {datasets}")
    print(f"  Models: {models}")
    print(f"  Seeds: {seeds}")
    print(f"  Total experiments: {total_exp}")
    print(f"  Epochs: {epochs}")
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
                print(f"# Experiment {exp_idx}/{total_exp}: {dataset_name} + {model_name} (seed={seed})")
                print(f"{'#'*70}")
                
                result = run_single_experiment_with_seed(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose
                )
                
                all_results.append(result)
                
                # Save intermediate results
                _save_intermediate(all_results, save_dir, timestamp)
    
    total_time = time.time() - total_start_time
    
    # Save full results
    results_df = save_results_multi_seed(all_results, save_dir, timestamp)
    
    # Print summary
    print_summary_multi_seed(results_df, seeds, total_time)
    
    return results_df


def _save_intermediate(results: list, save_dir: str, timestamp: str):
    """Save intermediate results"""
    json_path = os.path.join(save_dir, f'results_multi_seed_{timestamp}_intermediate.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


def save_results_multi_seed(results: list, save_dir: str, timestamp: str) -> pd.DataFrame:
    """Save multi-seed results"""
    # 1. Save full JSON
    json_path = os.path.join(save_dir, f'results_multi_seed_{timestamp}.json')
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
            }
        else:
            row = {
                'Dataset': r['dataset'],
                'Model': r['model'],
                'Seed': r['seed'],
                'ACC': 0, 'AUC': 0, 'F1': 0, 'AP': 0,
            }
        rows.append(row)
    
    results_df = pd.DataFrame(rows)
    
    # 3. Save raw CSV
    csv_path = os.path.join(save_dir, f'results_multi_seed_{timestamp}_raw.csv')
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
    summary_csv_path = os.path.join(save_dir, f'results_multi_seed_{timestamp}_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Summary CSV: {summary_csv_path}")
    
    # 5. Remove intermediate files
    intermediate_path = os.path.join(save_dir, f'results_multi_seed_{timestamp}_intermediate.json')
    if os.path.exists(intermediate_path):
        os.remove(intermediate_path)
    
    return summary_df


def print_summary_multi_seed(summary_df: pd.DataFrame, seeds: list, total_time: float):
    """Print multi-seed summary"""
    print("\n" + "=" * 90)
    print(f"SUMMARY (mean ± std over {len(seeds)} seeds)")
    print("=" * 90)
    
    datasets = ['bus', 'fundus', 'glaucoma', 'mammo_calc', 'mammo_mass']
    models = ['imagenet_vit', 'ours', 'radimagenet_resnet50', 'radimagenet_densenet121', 'radimagenet_inceptionv3']
    
    # ACC table
    print("\n[ACC (%) - mean ± std]")
    print(f"{'Model':<28} | {'bus':>12} | {'fundus':>12} | {'glaucoma':>12} | {'mammo_calc':>12} | {'mammo_mass':>12} | {'Average':>12}")
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
        print(f"{model:<28} | {row_data[0]:>12} | {row_data[1]:>12} | {row_data[2]:>12} | {row_data[3]:>12} | {row_data[4]:>12} | {avg_mean:>10.2f}")
    
    # AUC table
    print("\n[AUC (%) - mean ± std]")
    print(f"{'Model':<28} | {'bus':>12} | {'fundus':>12} | {'glaucoma':>12} | {'mammo_calc':>12} | {'mammo_mass':>12} | {'Average':>12}")
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
        print(f"{model:<28} | {row_data[0]:>12} | {row_data[1]:>12} | {row_data[2]:>12} | {row_data[3]:>12} | {row_data[4]:>12} | {avg_mean:>10.2f}")
    
    print(f"\n[Total Time] {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description='External Validation - Multi-Seed')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to evaluate (default: all)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to evaluate (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int, default=RANDOM_SEEDS,
                        help='Random seeds (default: 42, 123, 456)')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'],
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--save_dir', type=str, default=RESULTS_ROOT,
                        help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')
    
    args = parser.parse_args()
    
    results_df = run_all_experiments_multi_seed(
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
        # Quick test: 1 dataset, 1 model, 2 seeds, 5 epochs
        print("=" * 70)
        print("Quick Test (1 dataset, 1 model, 2 seeds, 5 epochs)")
        print("=" * 70)
        
        results_df = run_all_experiments_multi_seed(
            datasets=['glaucoma'],
            models=['imagenet_vit'],
            seeds=[42, 123],
            epochs=5,
            verbose=True
        )
        
        print("\n[Test completed]")
        print(results_df)
    else:
        main()