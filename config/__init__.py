# config/__init__.py
from .datasets import (
    DatasetConfig, DATASET_CONFIGS,
    DATASETS_2D, DATASETS_3D, ALL_DATASETS, INTERLEAVED_DATASETS,
    EXPERT_ROUTING, EXPERT_BOTTLENECK,
    get_dataset_config, get_num_classes_list, get_task_id, get_expert_id, is_3d_dataset,
)