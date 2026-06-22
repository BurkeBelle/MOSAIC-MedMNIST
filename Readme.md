# MOSAIC: Mixture-of-Specialists Adapter for Imaging Classification

> **MOSAIC: A Unified Parameter-Efficient Framework for Heterogeneous Medical Image Classification**
>
> Submitted to WACV 2027

## Overview

MOSAIC addresses **cross-task representation interference** when unifying heterogeneous medical imaging datasets. It combines:

- **Frozen ViT-Base backbone** (ImageNet pretrained)
- **MedCoSS-style Tokenizer** for unified 2D/3D input processing
- **Mixture-of-Specialists (MoS) Adapters** with hard routing for modality-specific adaptation
- **Cyclic Teacher-Student training** (Ark+ style) for multi-dataset learning

The system trains on **18 MedMNIST datasets** (12 2D + 6 3D) with only **~7.9% trainable parameters**.

## Architecture

```
Input (2D/3D) → Tokenizer → [CLS] + Patches + PosEmbed
                                    ↓
              Frozen ViT-Base Transformer × 12 layers
              (each layer has parallel MoS Adapter)
                                    ↓
                        CLS Feature → Task Head → Prediction
```

**Three-Expert Hard Routing:**
| Expert | Modality | Datasets | Bottleneck |
|--------|----------|----------|------------|
| A (Bio-Medical) | RGB color, microscopic texture | PathMNIST, BloodMNIST, DermaMNIST, ... | 64 |
| B (Radiology) | Grayscale, macro geometry | ChestMNIST, PneumoniaMNIST, OCTMNIST, ... | 96 |
| C (Volumetric) | 3D voxel, spatial structure | OrganMNIST3D, NoduleMNIST3D, ... | 128 |

## Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/MOSAIC.git
cd MOSAIC

# Environment
conda create -n mosaic python=3.9 -y
conda activate mosaic
pip install -r requirements.txt

# Download ViT-Base pretrained weights
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -O vit_base_patch16_224.npz
```

## Data Preparation

Download MedMNIST datasets (224×224 for 2D, 64×64×64 for 3D):

```python
import medmnist
# 2D datasets (size=224)
for ds in ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
           'chestmnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
           'retinamnist', 'organamnist', 'organcmnist', 'organsmnist']:
    medmnist.INFO[ds]  # downloads automatically on first use

# 3D datasets (size=64)
for ds in ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
           'vesselmnist3d', 'fracturemnist3d', 'synapsemnist3d']:
    medmnist.INFO[ds]
```

Place all data under `./data_224/` (or specify via `--data_root`).

## Training

### MOSAIC (Proposed Method)

```bash
# V2: MoE three-expert adapter (default)
python train.py \
    --data_root ./data_224 \
    --pretrained ./vit_base_patch16_224.npz \
    --adapter_mode v2_moe \
    --freeze_backbone \
    --num_rounds 72 \
    --batch_size 32 \
    --lr 1e-4 \
    --seed 42 \
    --output_dir ./output/mosaic_seed42
```

### Baselines (LoRA / VPT)

```bash
# LoRA Joint (shared LoRA, cyclic training)
python train_baseline.py --mode joint --adapter lora --seed 42

# VPT Independent (18 separate models)
python train_baseline.py --mode independent --adapter vpt --seed 42

# Run all 4 baseline experiments × 3 seeds
python train_baseline.py --run_all --seeds 42 123 456
```

## Results

Results on 18 MedMNIST datasets (mean ACC ± std over 3 seeds):

| Method | Mode | Mean ACC | 2D ACC | 3D ACC | Trainable Params |
|--------|------|----------|--------|--------|-----------------|
| LoRA (r=8) | Independent | - | - | - | 0.31M |
| LoRA (r=8) | Joint | - | - | - | 0.31M |
| VPT (P=10) | Independent | - | - | - | 0.11M |
| VPT (P=10) | Joint | - | - | - | 0.11M |
| **MOSAIC (Ours)** | Joint | **84.16** | **88.92** | **72.73** | **6.84M** |

## Project Structure

```
MOSAIC/
├── train.py                  # MOSAIC training entry
├── train_baseline.py         # Baseline experiments (LoRA/VPT)
├── test.py                   # Evaluation script
├── config/datasets.py        # Dataset configs & expert routing
├── dataloader/
│   ├── transforms.py         # 2D/3D augmentations
│   └── medmnist_loader.py    # MedMNIST data loading
├── model/
│   ├── adapter.py            # AdaptFormer & MoE Adapter
│   ├── patch_embed.py        # 2D/3D Patch Embedding
│   ├── transformer_block.py  # ViT Block with Adapter
│   ├── unified_model.py      # MOSAIC model & Teacher
│   ├── lora_adapter.py       # LoRA baseline
│   ├── vpt_adapter.py        # VPT-Deep baseline
│   └── baseline_model.py     # Baseline model factory
├── engine/
│   ├── trainer.py            # Cyclic training loop
│   └── evaluator.py          # Multi-metric evaluation
└── utils/logger.py           # Experiment logging
```

## Citation

```bibtex
@inproceedings{mosaic2027,
  title={MOSAIC: A Unified Parameter-Efficient Framework for Heterogeneous Medical Image Classification},
  author={},
  booktitle={Proceedings of WACV},
  year={2027}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- [MedMNIST](https://medmnist.com/) for the benchmark datasets
- [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer) for the adapter architecture
- [Ark+](https://github.com/JLiangLab/Ark) for the cyclic training strategy
- [MedCoSS](https://github.com/yeerwen/MedCoSS) for the multi-modal tokenizer design