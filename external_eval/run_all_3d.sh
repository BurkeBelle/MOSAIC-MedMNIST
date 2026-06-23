#!/bin/bash
# Run all 3D models on COVID-CT dataset
# Includes both Linear Probe and Full Fine-tuning modes
# 10 seeds
# Models: UniMiSS, ImageNet ViT, Med3D (ResNet18/34/50), Ours

set -e

SEEDS=(42 123 456 789 1024 2048 3407 4096 5000 6666)
OUTPUT_PROBE="results_3d_probe_10seeds"
OUTPUT_FINETUNE="results_3d_finetune_10seeds"

mkdir -p $OUTPUT_PROBE
mkdir -p $OUTPUT_FINETUNE

echo "=========================================="
echo "3D External Validation - All Models"
echo "=========================================="
echo "Seeds: ${SEEDS[@]}"
echo "Total seeds: ${#SEEDS[@]}"
echo "Models: unimiss, imagenet_vit, med3d_resnet18, med3d_resnet34, med3d_resnet50, ours"
echo ""

# ==========================================
# Part 1: Linear Probe (backbone frozen)
# ==========================================
echo ""
echo "############################################"
echo "# Part 1: Linear Probe (10 seeds)"
echo "############################################"

# 1. UniMiSS - Probe
echo ""
echo "[Probe] UniMiSS"
for seed in "${SEEDS[@]}"; do
    echo ">>> UniMiSS Probe, Seed: $seed"
    python train_unimiss_3d.py \
        --model unimiss \
        --seed $seed \
        --epochs 50 \
        --batch_size 8 \
        --lr_head 1e-3 \
        --freeze_backbone \
        --output_dir $OUTPUT_PROBE
done

# 2. ImageNet ViT - Probe
echo ""
echo "[Probe] ImageNet ViT"
for seed in "${SEEDS[@]}"; do
    echo ">>> ImageNet ViT Probe, Seed: $seed"
    python train_unimiss_3d.py \
        --model imagenet_vit \
        --seed $seed \
        --epochs 50 \
        --batch_size 8 \
        --lr_head 1e-3 \
        --freeze_backbone \
        --output_dir $OUTPUT_PROBE
done

# 3. Med3D ResNet-18 - Probe
echo ""
echo "[Probe] Med3D ResNet-18"
for seed in "${SEEDS[@]}"; do
    echo ">>> Med3D ResNet-18 Probe, Seed: $seed"
    python train_unimiss_3d.py \
        --model med3d_resnet18 \
        --seed $seed \
        --epochs 50 \
        --batch_size 16 \
        --lr_head 1e-3 \
        --freeze_backbone \
        --output_dir $OUTPUT_PROBE
done

# 4. Med3D ResNet-34 - Probe
echo ""
echo "[Probe] Med3D ResNet-34"
for seed in "${SEEDS[@]}"; do
    echo ">>> Med3D ResNet-34 Probe, Seed: $seed"
    python train_unimiss_3d.py \
        --model med3d_resnet34 \
        --seed $seed \
        --epochs 50 \
        --batch_size 16 \
        --lr_head 1e-3 \
        --freeze_backbone \
        --output_dir $OUTPUT_PROBE
done

# 5. Med3D ResNet-50 - Probe
echo ""
echo "[Probe] Med3D ResNet-50"
for seed in "${SEEDS[@]}"; do
    echo ">>> Med3D ResNet-50 Probe, Seed: $seed"
    python train_unimiss_3d.py \
        --model med3d_resnet50 \
        --seed $seed \
        --epochs 50 \
        --batch_size 8 \
        --lr_head 1e-3 \
        --freeze_backbone \
        --output_dir $OUTPUT_PROBE
done

# 6. Ours - Probe
echo ""
echo "[Probe] Ours (Expert C)"
for seed in "${SEEDS[@]}"; do
    echo ">>> Ours Probe, Seed: $seed"
    python train_unimiss_3d.py \
        --model ours \
        --seed $seed \
        --epochs 50 \
        --batch_size 16 \
        --lr_head 1e-3 \
        --freeze_backbone \
        --output_dir $OUTPUT_PROBE
done

# ==========================================
# Part 2: Fine-tuning (Baselines: Full FT, Ours: Adapter Tuning)
# ==========================================
echo ""
echo "############################################"
echo "# Part 2: Fine-tuning (10 seeds)"
echo "# Baselines: Full Fine-tuning"
echo "# Ours: Adapter Tuning"
echo "############################################"

# 1. UniMiSS - Finetune
echo ""
echo "[Finetune] UniMiSS (Full FT)"
for seed in "${SEEDS[@]}"; do
    echo ">>> UniMiSS Finetune, Seed: $seed"
    python train_unimiss_3d.py \
        --model unimiss \
        --seed $seed \
        --epochs 50 \
        --batch_size 4 \
        --lr 1e-5 \
        --lr_head 1e-3 \
        --output_dir $OUTPUT_FINETUNE
done

# 2. ImageNet ViT - Finetune
echo ""
echo "[Finetune] ImageNet ViT (Full FT)"
for seed in "${SEEDS[@]}"; do
    echo ">>> ImageNet ViT Finetune, Seed: $seed"
    python train_unimiss_3d.py \
        --model imagenet_vit \
        --seed $seed \
        --epochs 50 \
        --batch_size 4 \
        --lr 1e-5 \
        --lr_head 1e-3 \
        --output_dir $OUTPUT_FINETUNE
done

# 3. Med3D ResNet-18 - Finetune
echo ""
echo "[Finetune] Med3D ResNet-18 (Full FT)"
for seed in "${SEEDS[@]}"; do
    echo ">>> Med3D ResNet-18 Finetune, Seed: $seed"
    python train_unimiss_3d.py \
        --model med3d_resnet18 \
        --seed $seed \
        --epochs 50 \
        --batch_size 8 \
        --lr 1e-4 \
        --lr_head 1e-3 \
        --output_dir $OUTPUT_FINETUNE
done

# 4. Med3D ResNet-34 - Finetune
echo ""
echo "[Finetune] Med3D ResNet-34 (Full FT)"
for seed in "${SEEDS[@]}"; do
    echo ">>> Med3D ResNet-34 Finetune, Seed: $seed"
    python train_unimiss_3d.py \
        --model med3d_resnet34 \
        --seed $seed \
        --epochs 50 \
        --batch_size 8 \
        --lr 1e-4 \
        --lr_head 1e-3 \
        --output_dir $OUTPUT_FINETUNE
done

# 5. Med3D ResNet-50 - Finetune
echo ""
echo "[Finetune] Med3D ResNet-50 (Full FT)"
for seed in "${SEEDS[@]}"; do
    echo ">>> Med3D ResNet-50 Finetune, Seed: $seed"
    python train_unimiss_3d.py \
        --model med3d_resnet50 \
        --seed $seed \
        --epochs 50 \
        --batch_size 4 \
        --lr 1e-4 \
        --lr_head 1e-3 \
        --output_dir $OUTPUT_FINETUNE
done

# 6. Ours - Adapter Tuning
echo ""
echo "[Finetune] Ours (Adapter Tuning)"
for seed in "${SEEDS[@]}"; do
    echo ">>> Ours Adapter Tuning, Seed: $seed"
    python train_unimiss_3d.py \
        --model ours \
        --seed $seed \
        --epochs 50 \
        --batch_size 8 \
        --lr 1e-4 \
        --lr_head 1e-3 \
        --output_dir $OUTPUT_FINETUNE
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

models = ['unimiss', 'imagenet_vit', 'med3d_resnet18', 'med3d_resnet34', 'med3d_resnet50', 'ours']
seeds = [42, 123, 456, 789, 1024, 2048, 3407, 4096, 5000, 6666]

for mode, output_dir in [('Linear Probe', '$OUTPUT_PROBE'), ('Fine-tuning', '$OUTPUT_FINETUNE')]:
    print('\\n' + '='*70)
    print(f'COVID-CT 3D Results - {mode} (10 seeds)')
    print('='*70)
    print(f'{\"Model\":<20} {\"ACC\":<15} {\"AUC\":<15} {\"F1\":<15} {\"AP\":<15}')
    print('-'*70)
    
    for model in models:
        accs, aucs, f1s, aps = [], [], [], []
        for seed in seeds:
            result_file = os.path.join(output_dir, f'{model}_covid_seed{seed}.json')
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
            print(f'{model:<20} {acc_str:<15} {auc_str:<15} {f1_str:<15} {ap_str:<15}')
        else:
            print(f'{model:<20} No results found')
    
    print('='*70)
"

echo ""
echo "All experiments completed!"
echo "Probe results: $OUTPUT_PROBE"
echo "Finetune results: $OUTPUT_FINETUNE"