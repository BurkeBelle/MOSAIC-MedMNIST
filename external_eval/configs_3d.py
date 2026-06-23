# external_eval/configs_3d.py
"""
3D External Validation configuration

Dataset: MosMedData (COVID-19 CT)
Models: ImageNet ViT (3D), Med3D ResNet-18/34/50, Ours (Expert C)
"""

import os
import torch

# ============================================================
# Path configuration
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'Data_MosMedData')
WEIGHTS_ROOT = os.path.join(os.path.dirname(__file__), 'weights')
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results_3d')

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seeds (same as 2D)
RANDOM_SEEDS = [42, 123, 456]

# ============================================================
# Dataset configuration
# ============================================================
DATASET_3D = {
    'name': 'MosMedData COVID-19 CT',
    'num_classes': 2,
    'class_names': ['Normal', 'COVID'],
    'task_type': 'binary',
    'in_channels': 1,        # grayscale CT
    'volume_size': 64,       # 64×64×64
    'train_file': os.path.join(DATA_ROOT, 'train.npy'),
    'test_file': os.path.join(DATA_ROOT, 'test.npy'),
    'train_labels_file': os.path.join(DATA_ROOT, 'train_labels.npy'),
    'test_labels_file': os.path.join(DATA_ROOT, 'test_labels.npy'),
    'n_train': 888,
    'n_test': 222,
}

# ============================================================
# Model configuration
# ============================================================
MODELS_3D = {
    # --- General pretrained (2D ImageNet ViT counterpart) ---
    'imagenet_vit_3d': {
        'display_name': 'ImageNet ViT-B',
        'arch': 'unified_model',
        'weights': os.path.join(WEIGHTS_ROOT, 'imagenet_vit.npz'),
        'pretrain_data': 'ImageNet-1K (2D)',
        'description': 'ViT-Base backbone from ImageNet, 3D tokenizer randomly initialized',
        'embed_dim': 768,
    },
    # --- 3D medical pretrained (2D RadImageNet counterpart) ---
    'med3d_resnet18': {
        'display_name': 'Med3D ResNet-18',
        'arch': 'resnet3d',
        'model_depth': 18,
        'resnet_shortcut': 'A',
        'weights': os.path.join(WEIGHTS_ROOT, 'pretrain', 'resnet_18.pth'),
        'pretrain_data': '8 Medical 3D Datasets',
        'description': '3D-ResNet-18 pretrained on 8 CT/MRI datasets',
        'embed_dim': 512,
    },
    'med3d_resnet34': {
        'display_name': 'Med3D ResNet-34',
        'arch': 'resnet3d',
        'model_depth': 34,
        'resnet_shortcut': 'A',
        'weights': os.path.join(WEIGHTS_ROOT, 'pretrain', 'resnet_34.pth'),
        'pretrain_data': '8 Medical 3D Datasets',
        'description': '3D-ResNet-34 pretrained on 8 CT/MRI datasets',
        'embed_dim': 512,
    },
    'med3d_resnet50': {
        'display_name': 'Med3D ResNet-50',
        'arch': 'resnet3d',
        'model_depth': 50,
        'resnet_shortcut': 'B',
        'weights': os.path.join(WEIGHTS_ROOT, 'pretrain', 'resnet_50.pth'),
        'pretrain_data': '8 Medical 3D Datasets',
        'description': '3D-ResNet-50 pretrained on 8 CT/MRI datasets',
        'embed_dim': 2048,
    },
    # --- Ours ---
    'ours_expert_c': {
        'display_name': 'Ours (Expert C)',
        'arch': 'unified_model_moe',
        'weights': os.path.join(WEIGHTS_ROOT, 'ours.pth'),
        'pretrain_data': 'MedMNIST 3D × 6 datasets',
        'description': 'UnifiedModel + MoE Expert C for 3D volumetric data',
        'embed_dim': 768,
    },
}

# ============================================================
# Training hyperparameters (same as 2D)
# ============================================================
TRAIN_CONFIG = {
    'optimizer': 'adam',
    'lr': 0.001,
    'weight_decay': 1e-4,
    'epochs': 50,
    'batch_size': 16,         # 3D data is memory-intensive, use smaller batch
    'num_workers': 4,
    'scheduler': 'cosine',
    'early_stopping': True,
    'patience': 10,
}

# ============================================================
# Utility functions
# ============================================================
def get_model_config(model_name: str) -> dict:
    """Get model configuration"""
    if model_name not in MODELS_3D:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(MODELS_3D.keys())}")
    return MODELS_3D[model_name]


def get_available_models() -> list:
    """Get all available model names"""
    return list(MODELS_3D.keys())


def print_config():
    """Print configuration"""
    print("\n" + "=" * 60)
    print("3D External Validation Configuration")
    print("=" * 60)

    print("\n[Paths]")
    print(f"  Project Root:  {PROJECT_ROOT}")
    print(f"  Data Root:     {DATA_ROOT}")
    print(f"  Weights Root:  {WEIGHTS_ROOT}")
    print(f"  Results Root:  {RESULTS_ROOT}")

    print(f"\n[Dataset]")
    d = DATASET_3D
    print(f"  {d['name']}: {d['num_classes']} classes ({d['class_names']})")
    print(f"  Volume: {d['volume_size']}³, Channels: {d['in_channels']}")
    print(f"  Train: {d['n_train']}, Test: {d['n_test']}")

    print("\n[Models]")
    for name, cfg in MODELS_3D.items():
        status = "✓" if os.path.exists(cfg['weights']) else "✗"
        print(f"  [{status}] {name}: {cfg['display_name']} ({cfg['arch']})")
        print(f"       Pretrain: {cfg['pretrain_data']}")

    print("\n[Training]")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n[Seeds]")
    print(f"  {RANDOM_SEEDS}")

    print("\n" + "=" * 60)


# ============================================================
if __name__ == "__main__":
    print_config()