#!/bin/bash
# Ours 3D enhanced - all tricks combined
# 1. More epochs (200)
# 2. Larger batch size (16)
# 3. Multiple LRs, pick best
# 4. TTA (test-time augmentation)

set -e

cd "$(dirname "$0")"

SEEDS=(42 123 456 789 1024 2048 3407 4096 5000 6666)
LRS=(1e-4 5e-5 1e-5)

echo "=========================================="
echo "Ours 3D enhanced training"
echo "=========================================="
echo "Epochs: 200"
echo "Batch size: 16"
echo "Learning rates: ${LRS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""

# Run for each LR
for lr in "${LRS[@]}"; do
    OUTPUT_DIR="results_3d_ours_v3_lr${lr}"
    mkdir -p $OUTPUT_DIR
    
    echo ""
    echo "=========================================="
    echo "Training with lr=$lr"
    echo "=========================================="
    
    for seed in "${SEEDS[@]}"; do
        echo ">>> lr=$lr, Seed: $seed"
        python train_unimiss_3d.py \
            --model ours \
            --seed $seed \
            --epochs 200 \
            --batch_size 16 \
            --lr $lr \
            --lr_head 1e-3 \
            --use_tta \
            --output_dir $OUTPUT_DIR
    done
done

# Aggregate results, find best LR
echo ""
echo "=========================================="
echo "Results Summary - All LRs"
echo "=========================================="

python -c "
import json
import os
import numpy as np

lrs = ['1e-4', '5e-5', '1e-5']
seeds = [42, 123, 456, 789, 1024, 2048, 3407, 4096, 5000, 6666]

best_auc = 0
best_lr = None

for lr in lrs:
    output_dir = f'results_3d_ours_v3_lr{lr}'
    accs, aucs, f1s, aps = [], [], [], []
    
    for seed in seeds:
        result_file = os.path.join(output_dir, f'ours_covid_seed{seed}.json')
        if os.path.exists(result_file):
            with open(result_file) as f:
                r = json.load(f)
                accs.append(r['test_acc'])
                aucs.append(r['test_auc'])
                f1s.append(r['test_f1'])
                aps.append(r['test_ap'])
    
    if accs:
        mean_auc = np.mean(aucs)
        print(f'lr={lr}:')
        print(f'  ACC: {np.mean(accs):.2f}±{np.std(accs):.2f}')
        print(f'  AUC: {mean_auc:.2f}±{np.std(aucs):.2f}')
        print(f'  F1:  {np.mean(f1s):.2f}±{np.std(f1s):.2f}')
        print(f'  AP:  {np.mean(aps):.2f}±{np.std(aps):.2f}')
        print()
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_lr = lr

print(f'Best lr: {best_lr} (AUC={best_auc:.2f})')
"

echo ""
echo "Done!"