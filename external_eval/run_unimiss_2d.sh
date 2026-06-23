#!/bin/bash
# UniMiSS 2D External Validation
# Linear Probe + Fine-tuning, 10 seeds

set -e

cd "$(dirname "$0")"

SEEDS=(42 123 456 789 1024 2048 3407 4096 5000 6666)
DATASETS=(bus fundus glaucoma mammo_calc mammo_mass)

OUTPUT_PROBE="results_unimiss_2d_probe"
OUTPUT_FINETUNE="results_unimiss_2d_finetune"

mkdir -p $OUTPUT_PROBE
mkdir -p $OUTPUT_FINETUNE

echo "=========================================="
echo "UniMiSS 2D External Validation"
echo "=========================================="
echo "Datasets: ${DATASETS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""

# ==========================================
# Part 1: Linear Probe
# ==========================================
echo ""
echo "############################################"
echo "# Part 1: UniMiSS Linear Probe"
echo "############################################"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "[Probe] UniMiSS - $dataset"
    for seed in "${SEEDS[@]}"; do
        echo ">>> $dataset, Seed: $seed"
        python train_unimiss_2d.py \
            --dataset $dataset \
            --seed $seed \
            --epochs 50 \
            --batch_size 32 \
            --lr_head 1e-3 \
            --freeze_backbone \
            --output_dir $OUTPUT_PROBE
    done
done

# ==========================================
# Part 2: Fine-tuning (Full FT for UniMiSS)
# ==========================================
echo ""
echo "############################################"
echo "# Part 2: UniMiSS Fine-tuning"
echo "############################################"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "[Finetune] UniMiSS - $dataset"
    for seed in "${SEEDS[@]}"; do
        echo ">>> $dataset, Seed: $seed"
        python train_unimiss_2d.py \
            --dataset $dataset \
            --seed $seed \
            --epochs 50 \
            --batch_size 16 \
            --lr 1e-5 \
            --lr_head 1e-3 \
            --output_dir $OUTPUT_FINETUNE
    done
done

# ========================================
# Aggregate Results
# ========================================
echo ""
echo "=========================================="
echo "Aggregating Results..."
echo "=========================================="

python -c "
import json
import os
import numpy as np

datasets = ['bus', 'fundus', 'glaucoma', 'mammo_calc', 'mammo_mass']
seeds = [42, 123, 456, 789, 1024, 2048, 3407, 4096, 5000, 6666]

for mode, output_dir in [('Linear Probe', '$OUTPUT_PROBE'), ('Fine-tuning', '$OUTPUT_FINETUNE')]:
    print('\\n' + '='*80)
    print(f'UniMiSS 2D Results - {mode} (10 seeds)')
    print('='*80)
    print(f'{\"Dataset\":<15} {\"ACC\":<18} {\"AUC\":<18} {\"F1\":<18} {\"AP\":<18}')
    print('-'*80)
    
    all_aucs = []
    for dataset in datasets:
        accs, aucs, f1s, aps = [], [], [], []
        for seed in seeds:
            result_file = os.path.join(output_dir, f'unimiss_{dataset}_seed{seed}.json')
            if os.path.exists(result_file):
                with open(result_file) as f:
                    r = json.load(f)
                    accs.append(r['test_acc'])
                    aucs.append(r['test_auc'])
                    f1s.append(r['test_f1'])
                    aps.append(r['test_ap'])
        
        if accs:
            acc_str = f'{np.mean(accs):.2f}±{np.std(accs):.2f}'
            auc_str = f'{np.mean(aucs):.2f}±{np.std(aucs):.2f}'
            f1_str = f'{np.mean(f1s):.2f}±{np.std(f1s):.2f}'
            ap_str = f'{np.mean(aps):.2f}±{np.std(aps):.2f}'
            print(f'{dataset:<15} {acc_str:<18} {auc_str:<18} {f1_str:<18} {ap_str:<18}')
            all_aucs.extend(aucs)
        else:
            print(f'{dataset:<15} No results found')
    
    if all_aucs:
        print('-'*80)
        print(f'{\"Average\":<15} {\"\":<18} {np.mean(all_aucs):.2f}±{np.std(all_aucs):.2f}')
    print('='*80)
"

echo ""
echo "All UniMiSS 2D experiments completed!"
echo "Probe results: $OUTPUT_PROBE"
echo "Finetune results: $OUTPUT_FINETUNE"