# external_eval/configs.py
"""
External Validation configuration

Dataset info, model weights paths, and training hyperparameters.
"""

import os

# ============================================================
# Path configuration
# ============================================================
# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Dataset root (extracted MedIMeta location)
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'Data_MedIMeta')

# Weights directory
WEIGHTS_ROOT = os.path.join(PROJECT_ROOT, 'external_eval', 'weights')

# Results directory
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'external_eval', 'results')


# ============================================================
# Dataset configuration
# ============================================================
# Expert A: Bio-Medical RGB (pathology, dermoscopy, fundus)
# Expert B: Radiology Grayscale (X-ray, ultrasound, CT)
# Expert C: 3D Volumetric

DATASETS = {
    'bus': {
        'name': 'Breast Ultrasound',
        'task_name': 'case category',
        'task_type': 'multiclass',
        'num_classes': 3,
        'in_channels': 1,  # grayscale, converted to 3-channel
        'input_size': 224,
        'expert': 'B',  # Ultrasound -> Radiology
        'labels': {
            0: 'normal',
            1: 'benign',
            2: 'malignant'
        }
    },
    'fundus': {
        'name': 'Fundus Multi-disease',
        'task_name': 'disease presence',
        'task_type': 'binary',
        'num_classes': 2,
        'in_channels': 3,
        'input_size': 224,
        'expert': 'A',  # Fundus RGB -> Bio-Medical
        'labels': {
            0: 'normal',
            1: 'abnormal'
        }
    },
    'glaucoma': {
        'name': 'Glaucoma Detection',
        'task_name': 'Glaucoma suspect',
        'task_type': 'binary',
        'num_classes': 2,
        'in_channels': 3,
        'input_size': 224,
        'expert': 'A',  # Fundus RGB -> Bio-Medical
        'labels': {
            0: 'Normal',
            1: 'Suspect'
        }
    },
    'mammo_calc': {
        'name': 'Mammography (Calcifications)',
        'task_name': 'pathology',
        'task_type': 'binary',
        'num_classes': 2,
        'in_channels': 1,  # grayscale, converted to 3-channel
        'input_size': 224,
        'expert': 'B',  # Mammography -> Radiology
        'labels': {
            0: 'benign',
            1: 'malignant'
        }
    },
    'mammo_mass': {
        'name': 'Mammography (Masses)',
        'task_name': 'pathology',
        'task_type': 'binary',
        'num_classes': 2,
        'in_channels': 1,  # grayscale, converted to 3-channel
        'input_size': 224,
        'expert': 'B',  # Mammography -> Radiology
        'labels': {
            0: 'benign',
            1: 'malignant'
        }
    }
}


# ============================================================
# Model configuration
# ============================================================
MODELS = {
    'imagenet_vit': {
        'name': 'ImageNet ViT-Base',
        'arch': 'vit_base',
        'weights': os.path.join(WEIGHTS_ROOT, 'imagenet_vit.npz'),
        'embed_dim': 768,
    },
    'medcoss': {
        'name': 'MedCoSS',
        'arch': 'uni_perceiver',  # compatibility TBD
        'weights': os.path.join(WEIGHTS_ROOT, 'medcoss.pth'),
        'embed_dim': 768,
    },
    'radimagenet_resnet50': {
        'name': 'RadImageNet ResNet50',
        'arch': 'resnet50',
        'weights': os.path.join(WEIGHTS_ROOT, 'radimagenet_resnet50.pt'),
        'embed_dim': 2048,
    },
    'radimagenet_densenet121': {
        'name': 'RadImageNet DenseNet121',
        'arch': 'densenet121',
        'weights': os.path.join(WEIGHTS_ROOT, 'radimagenet_densenet121.pt'),
        'embed_dim': 1024,
    },
    'radimagenet_inceptionv3': {
        'name': 'RadImageNet InceptionV3',
        'arch': 'inceptionv3',
        'weights': os.path.join(WEIGHTS_ROOT, 'radimagenet_inceptionv3.pt'),
        'embed_dim': 2048,
    },
    'ours': {
        'name': 'Ours (ViT-Base + MoE)',
        'arch': 'vit_moe',
        'weights': os.path.join(WEIGHTS_ROOT, 'ours.pth'),
        'embed_dim': 768,
    }
}


# ============================================================
# Training hyperparameters (Linear Probe)
# ============================================================
TRAIN_CONFIG = {
    'optimizer': 'adam',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 50,
    'batch_size': 32,
    'num_workers': 4,
    'scheduler': 'cosine',  # 'cosine' or 'step' or None
    'early_stopping': True,
    'patience': 10,  # early stopping patience
}


# ============================================================
# Evaluation metrics
# ============================================================
METRICS = ['accuracy', 'auc', 'f1', 'ap']


# ============================================================
# Other settings
# ============================================================
RANDOM_SEEDS = [42, 123, 456]  # 3 seeds
DEVICE = 'cuda'  # 'cuda' or 'cpu'


# ============================================================
# Utility functions
# ============================================================
def get_dataset_path(dataset_name: str, split: str = 'train') -> dict:
    """
    Get dataset path
    
    Args:
        dataset_name: dataset name (bus, fundus, glaucoma, mammo_calc, mammo_mass)
        split: split (train, val, test)
    
    Returns:
        dict: dict with images_dir, split_file, label_file paths
    """
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)
    
    # Get task_name for label file
    task_name = DATASETS[dataset_name]['task_name']
    
    return {
        'images_dir': os.path.join(dataset_dir, 'images'),
        'split_file': os.path.join(dataset_dir, 'splits', f'{split}.txt'),
        'label_file': os.path.join(dataset_dir, 'task_labels', f'{task_name}.npy'),
    }


def get_model_config(model_name: str) -> dict:
    """
    Get model configuration
    
    Args:
        model_name: model name
    
    Returns:
        dict: model config
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name]


def print_config():
    """Print configuration"""
    print("\n" + "=" * 60)
    print("External Validation Configuration")
    print("=" * 60)
    
    print("\n[Paths]")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Root: {DATA_ROOT}")
    print(f"  Weights Root: {WEIGHTS_ROOT}")
    print(f"  Results Root: {RESULTS_ROOT}")
    
    print("\n[Datasets]")
    for name, cfg in DATASETS.items():
        print(f"  {name}: {cfg['num_classes']} classes, {cfg['task_type']}")
    
    print("\n[Models]")
    for name, cfg in MODELS.items():
        print(f"  {name}: {cfg['arch']}")
    
    print("\n[Training]")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print_config()
    
    # Test path lookup
    print("\n[Test: get_dataset_path]")
    paths = get_dataset_path('glaucoma', 'train')
    for key, value in paths.items():
        print(f"  {key}: {value}")
        print(f"    exists: {os.path.exists(value)}")