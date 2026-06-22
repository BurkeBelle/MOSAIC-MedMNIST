# MOSAIC-MedMNIST

**Bridging Heterogeneous Medical Datasets via Mixture-of-Specialists Adapters for Unified Medical Image Classification**

*Accepted at MICCAI 2026*

## Overview

MOSAIC is a parameter-efficient framework that trains a single unified model across 18 heterogeneous MedMNIST datasets (12 2D + 6 3D) spanning six medical imaging modalities. The key idea is to address **cross-task representation interference** — the performance degradation that occurs when naïvely training one model on diverse medical data — through deterministic expert routing.

The framework combines a frozen ViT-Base backbone with three hard-routed specialist adapters (Bio-Medical / Radiology / Volumetric), each with a tailored bottleneck capacity. A MedCoSS-style tokenizer handles unified 2D/3D input processing, and an Ark+-style cyclic teacher-student training loop mitigates catastrophic forgetting across datasets.

With only **~7.05M trainable parameters (~7.9%)**, MOSAIC achieves **84.16% ACC** and **89.63 AUC** across all 18 datasets, matching or surpassing single-task specialists on 13 of 18 datasets.

## Architecture

```
Input (2D/3D) → Tokenizer → [CLS] + Patches + PosEmbed
                                    ↓
              Frozen ViT-Base Transformer × 12 layers
              (each layer has a parallel MoS Adapter)
                                    ↓
                        CLS Feature → Task Head → Prediction
```

**Three-Expert Hard Routing:**

| Expert | Modality | Datasets | Bottleneck |
|--------|----------|----------|------------|
| A (Bio-Medical) | RGB, microscopic texture | PathMNIST, BloodMNIST, TissueMNIST, DermaMNIST, RetinaMNIST | 64 |
| B (Radiology) | Grayscale, macro geometry | ChestMNIST, PneumoniaMNIST, BreastMNIST, OCTMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST | 96 |
| C (Volumetric) | 3D voxel, spatial structure | OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, VesselMNIST3D, FractureMNIST3D, SynapseMNIST3D | 192 |

## Installation

```bash
git clone https://github.com/BurkeBelle/MOSAIC-MedMNIST.git
cd MOSAIC-MedMNIST

conda create -n mosaic python=3.9 -y
conda activate mosaic
pip install -r requirements.txt
```

## Data Preparation

Download all 18 MedMNIST datasets (224×224 for 2D, 64×64×64 for 3D):

```python
import medmnist
from medmnist import INFO

# 2D datasets
for name in ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
             'chestmnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
             'retinamnist', 'organamnist', 'organcmnist', 'organsmnist']:
    DataClass = getattr(medmnist, INFO[name]['python_class'])
    DataClass(split='train', download=True, root='./data_224', size=224, as_rgb=True)

# 3D datasets
for name in ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
             'vesselmnist3d', 'fracturemnist3d', 'synapsemnist3d']:
    DataClass = getattr(medmnist, INFO[name]['python_class'])
    DataClass(split='train', download=True, root='./data_224', size=64)
```

## Pretrained Weights

Download the ViT-Base (ImageNet-21k) pretrained weights:

```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -O vit_base_patch16_224.npz
```

## Training

### MOSAIC (proposed method)

```bash
python train.py \
    --data_root ./data_224 \
    --pretrained ./vit_base_patch16_224.npz \
    --adapter_mode v2_moe \
    --adapter_bottleneck_a 64 \
    --adapter_bottleneck_b 96 \
    --adapter_bottleneck_c 192 \
    --freeze_backbone \
    --num_rounds 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --ema_momentum 0.9 \
    --ema_momentum_3d 0.95 \
    --consist_weight 0.1 \
    --early_stopping_patience 5 \
    --seed 42 \
    --output_dir ./output/mosaic_seed42

# Reproduce paper results (3 seeds)
for SEED in 42 123 456; do
    python train.py \
        --data_root ./data_224 \
        --pretrained ./vit_base_patch16_224.npz \
        --adapter_mode v2_moe \
        --adapter_bottleneck_a 64 \
        --adapter_bottleneck_b 96 \
        --adapter_bottleneck_c 192 \
        --freeze_backbone \
        --num_rounds 50 \
        --batch_size 32 \
        --lr 1e-4 \
        --ema_momentum 0.9 \
        --ema_momentum_3d 0.95 \
        --consist_weight 0.1 \
        --early_stopping_patience 5 \
        --seed $SEED \
        --output_dir ./output/mosaic_seed${SEED}
done
```

### Ablation: V1 dual-channel adapter

```bash
python train.py \
    --adapter_mode v1 \
    --freeze_backbone \
    --pretrained ./vit_base_patch16_224.npz \
    --num_rounds 50 \
    --ema_momentum 0.9 \
    --ema_momentum_3d 0.95 \
    --output_dir ./output/v1_seed42
```

### Ablation: 2D-3D interleaved training order

```bash
python train.py \
    --adapter_mode v2_moe \
    --adapter_bottleneck_c 192 \
    --use_interleaved \
    --freeze_backbone \
    --pretrained ./vit_base_patch16_224.npz \
    --num_rounds 50 \
    --ema_momentum 0.9 \
    --ema_momentum_3d 0.95 \
    --output_dir ./output/interleaved_seed42
```

### PEFT Baselines (LoRA / VPT)

```bash
# Run all 4 configurations (LoRA/VPT × Independent/Joint) with one seed
python train_baseline.py \
    --run_all --seeds 42 \
    --data_root ./data_224 \
    --pretrained ./vit_base_patch16_224.npz \
    --output_dir ./output_baseline

# Run all 4 configurations × 3 seeds (full reproduction)
python train_baseline.py \
    --run_all --seeds 42 123 456 \
    --data_root ./data_224 \
    --pretrained ./vit_base_patch16_224.npz \
    --output_dir ./output_baseline

# LoRA rank sweep (matched-budget comparison)
python train_baseline.py --mode joint --adapter lora \
    --lora_rank 48 --lora_alpha 96 --seed 42 \
    --data_root ./data_224 --pretrained ./vit_base_patch16_224.npz \
    --output_dir ./output_lora_r48

python train_baseline.py --mode joint --adapter lora \
    --lora_rank 192 --lora_alpha 384 --seed 42 \
    --data_root ./data_224 --pretrained ./vit_base_patch16_224.npz \
    --output_dir ./output_lora_r192
```

## Evaluation

```bash
python test.py --checkpoint ./output/mosaic_seed42/best_model.pth --data_root ./data_224
```

## Implementation Details

| Category | Parameter | Value |
|----------|-----------|-------|
| **Backbone** | Model | ViT-Base/16 |
| | Pretrained | ImageNet-21k (.npz) |
| | Freeze | ✓ |
| **Adapter** | Mode | v2_moe (3 experts) |
| | Expert A (Bio-Medical) | bottleneck = 64 |
| | Expert B (Radiology) | bottleneck = 96 |
| | Expert C (Volumetric) | bottleneck = 192 |
| | Scale | 0.1 |
| **EMA** | 2D Momentum | 0.9 |
| | 3D Momentum | 0.95 |
| **Loss** | Consistency Weight | 0.1 |
| | Total Loss | L_cls + 0.1 × L_consist |
| **Training** | Rounds | 50 (early stopping) |
| | Batch Size | 32 |
| | Learning Rate | 1e-4 |
| | Optimizer | AdamW (weight_decay=0.01) |
| | LR Schedule | Linear warmup (5 rounds) + Cosine |
| | Early Stopping | 5 rounds patience |
| **Seeds** | | 42, 123, 456 |

**Trainable Parameter Breakdown:**

| Component | Params |
|-----------|--------|
| 3D Tokenizer | 0.40M |
| Expert A × 12 layers | 1.19M |
| Expert B × 12 layers | 1.78M |
| Expert C × 12 layers | 3.55M |
| 18 Classification Heads | 0.13M |
| **Total Trainable** | **~7.05M (7.9%)** |

**Per-Seed Results:**

| Seed | ACC (%) |
|------|---------|
| 42 | 84.46 |
| 123 | 83.97 |
| 456 | 84.05 |
| **Mean ± Std** | **84.16 ± 0.26** |

### PEFT Baseline Settings

All baselines share the same frozen ViT-Base/16 backbone, tokenizer, and evaluation protocol as MOSAIC.

**Independent mode** (18 separate models, one per dataset):

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 0.01 |
| Max Epochs | 50 |
| Early Stopping | patience = 10 |
| Scheduler | CosineAnnealingLR |
| Batch Size | 32 |
| Teacher-Student | None |

**Joint mode** (single shared model, cyclic training):

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Max Rounds | 72 |
| Early Stopping | patience = 5 |
| Warmup | 5 rounds (linear 0.01→1.0) |
| Scheduler | Linear warmup + Cosine |
| Batch Size | 32 |
| Consistency Loss | MSE, weight = 0.1 |
| EMA Momentum | 0.999 |
| Gradient Clipping | max_norm = 1.0 |

**LoRA configuration:** rank = 8, alpha = 16 (applied to Q and V projections). Rank sweep: r ∈ {8, 48, 192} with alpha = 2r.

**VPT configuration:** VPT-Deep with 10 prompt tokens per layer, independent across layers, trunc_normal init (std=0.02).

## Results

### Main Results (Table 1)

Performance on 18 MedMNIST datasets (mean ± std over 3 seeds):

| Method | ACC (%) | AUC (%) |
|--------|---------|---------|
| Official (ResNet-18) | 80.32 | 89.92 |
| Single-task (ViT-B) | 81.86±0.38 | 87.81±0.07 |
| Joint baseline (ViT-B) | 77.66±0.66 | 84.55±0.35 |
| **MOSAIC (Ours)** | **84.16±0.21** | **89.63±0.24** |

| Split | ACC (%) | AUC (%) |
|-------|---------|---------|
| 2D Average | 88.78±0.28 | 95.24±0.09 |
| 3D Average | 74.93±0.27 | 78.41±0.91 |

### Matched-Budget PEFT Comparison (Table 2)

| Method | Training | #Models | Trainable | ACC (%) | AUC (%) |
|--------|----------|---------|-----------|---------|---------|
| LoRA (r=8) | Independent | 18 | 5.40M | 85.50±0.05 | 88.75±0.34 |
| VPT (P=10) | Independent | 18 | 1.80M | 83.83±0.45 | 86.73±1.22 |
| VPT (P=10) | Joint | 1 | 0.18M | 78.01±0.24 | 85.45±0.48 |
| LoRA (r=8) | Joint | 1 | 0.38M | 80.02±0.42 | 86.69±0.60 |
| LoRA (r=192) | Joint | 1 | 7.08M | 81.01±0.25 | 87.89±0.73 |
| **MOSAIC (Ours)** | **Joint** | **1** | **7.40M** | **84.16±0.21** | **89.63±0.24** |

## Project Structure

```
MOSAIC-MedMNIST/
├── train.py                  # MOSAIC training entry point
├── train_baseline.py         # PEFT baseline experiments (LoRA / VPT)
├── test.py                   # Checkpoint evaluation
├── config/
│   └── datasets.py           # Dataset configs & expert routing table
├── dataloader/
│   ├── medmnist_loader.py    # MedMNIST data loading (2D & 3D)
│   └── transforms.py         # 2D/3D augmentations (intensity-only for 3D)
├── model/
│   ├── adapter.py            # AdaptFormer adapter & MoE adapter
│   ├── patch_embed.py        # Unified 2D/3D patch embedding
│   ├── transformer_block.py  # ViT block with parallel adapter
│   ├── unified_model.py      # MOSAIC model & teacher (EMA)
│   ├── lora_adapter.py       # LoRA baseline
│   ├── vpt_adapter.py        # VPT-Deep baseline
│   └── baseline_model.py     # Baseline model factory
├── engine/
│   ├── trainer.py            # Cyclic training loop (Ark+ style)
│   └── evaluator.py          # Multi-metric evaluation (ACC, AUC)
├── utils/
│   └── logger.py             # Experiment logging & visualization
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{huang2026mosaic,
  title={Bridging Heterogeneous Medical Datasets via Mixture-of-Specialists Adapters for Unified Medical Image Classification},
  author={Huang, Shixing},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2026}
}
```

## Acknowledgements

- [MedMNIST](https://medmnist.com/) for the benchmark datasets
- [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer) for the adapter architecture
- [Ark](https://github.com/JLiangLab/Ark) for the cyclic training strategy
- [MedCoSS](https://github.com/yeerwen/MedCoSS) for the multi-modal tokenizer design

## License

Apache License 2.0
