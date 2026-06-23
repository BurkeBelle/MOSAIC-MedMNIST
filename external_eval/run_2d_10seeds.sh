#!/bin/bash
# 2D External Validation - 10 Seeds
# Linear Probe + Full Fine-tuning

set -e

cd "$(dirname "$0")"

# 10 seeds
SEEDS="42 123 456 789 1024 2048 3407 4096 5000 6666"

echo "=========================================="
echo "2D External Validation - 10 Seeds"
echo "=========================================="
echo "Seeds: $SEEDS"
echo ""

# ==========================================
# Part 1: Linear Probe (10 seeds)
# ==========================================
echo ""
echo "############################################"
echo "# Part 1: Linear Probe (10 seeds)"
echo "############################################"

python run_multi_seed.py \
    --seeds $SEEDS \
    --epochs 50 \
    --save_dir results_2d_probe_10seeds

# ==========================================
# Part 2: Full Fine-tuning (10 seeds)
# ==========================================
echo ""
echo "############################################"
echo "# Part 2: Full Fine-tuning (10 seeds)"
echo "############################################"

python run_finetune.py \
    --seeds $SEEDS \
    --epochs 100 \
    --save_dir results_2d_finetune_10seeds

echo ""
echo "=========================================="
echo "All 2D experiments completed!"
echo "=========================================="
echo "Probe results: results_2d_probe_10seeds/"
echo "Finetune results: results_2d_finetune_10seeds/"