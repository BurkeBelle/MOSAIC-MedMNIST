#!/bin/bash
# UniMiSS External Validation - Batch Run Script
# Run all 2D and 3D external validation experiments

set -e

# Seeds
SEEDS=(42 123 456)

# 2D Datasets
DATASETS_2D=(bus fundus glaucoma mammo_calc mammo_mass)

# Output directories
OUTPUT_2D="results_unimiss_2d"
OUTPUT_3D="results_unimiss_3d"

echo "=========================================="
echo "UniMiSS External Validation Experiments"
echo "=========================================="

# Run 2D experiments
echo ""
echo "Running 2D experiments..."
echo "=========================================="

for dataset in "${DATASETS_2D[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Dataset: $dataset, Seed: $seed"
        python train_unimiss_2d.py \
            --dataset $dataset \
            --seed $seed \
            --epochs 50 \
            --batch_size 32 \
            --lr 1e-4 \
            --lr_head 1e-3 \
            --output_dir $OUTPUT_2D
    done
done

# Run 3D experiments
echo ""
echo "Running 3D experiments (COVID-CT)..."
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> COVID-CT, Seed: $seed"
    python train_unimiss_3d.py \
        --seed $seed \
        --epochs 50 \
        --batch_size 4 \
        --lr 1e-5 \
        --lr_head 1e-3 \
        --volume_size 64 \
        --output_dir $OUTPUT_3D
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

# Aggregate results
echo ""
echo "Aggregating results..."
python -c "
import json
import os
import numpy as np

# 2D results
print('\\n=== 2D Results ===')
datasets = ['bus', 'fundus', 'glaucoma', 'mammo_calc', 'mammo_mass']
seeds = [42, 123, 456]

for dataset in datasets:
    aucs = []
    for seed in seeds:
        result_file = f'$OUTPUT_2D/unimiss_{dataset}_seed{seed}.json'
        if os.path.exists(result_file):
            with open(result_file) as f:
                results = json.load(f)
                aucs.append(results['best_auc'])
    if aucs:
        print(f'{dataset}: AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f}')

# 3D results
print('\\n=== 3D Results (COVID-CT) ===')
aucs = []
for seed in seeds:
    result_file = f'$OUTPUT_3D/unimiss_covid_seed{seed}.json'
    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)
            aucs.append(results['best_auc'])
if aucs:
    print(f'COVID-CT: AUC = {np.mean(aucs):.2f} ± {np.std(aucs):.2f}')
"