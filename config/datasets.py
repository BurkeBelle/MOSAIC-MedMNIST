# config/datasets.py
"""
MedMNIST dataset configuration.

Covers all 18 datasets (12 2D + 6 3D) with metadata and
three-expert hard-routing table for the MoE adapter.
"""

from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Per-dataset configuration."""
    name: str
    num_classes: int
    task_type: str          # multi-class | binary-class | multi-label | ordinal-regression
    is_3d: bool
    metric: str             # primary metric: 'acc' or 'auc'
    in_channels: int
    medmnist_name: str = ""

    def __post_init__(self):
        if not self.medmnist_name:
            self.medmnist_name = self.name.lower()


# ======================================================================
# Dataset registry
# ======================================================================

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    # --- 2D datasets (12) ---
    "PathMNIST":      DatasetConfig("PathMNIST",      9,  "multi-class",        False, "acc", 3, "pathmnist"),
    "DermaMNIST":     DatasetConfig("DermaMNIST",      7,  "multi-class",        False, "acc", 3, "dermamnist"),
    "OCTMNIST":       DatasetConfig("OCTMNIST",        4,  "multi-class",        False, "acc", 1, "octmnist"),
    "PneumoniaMNIST": DatasetConfig("PneumoniaMNIST",  2,  "binary-class",       False, "acc", 1, "pneumoniamnist"),
    "ChestMNIST":     DatasetConfig("ChestMNIST",      14, "multi-label",        False, "auc", 1, "chestmnist"),
    "BreastMNIST":    DatasetConfig("BreastMNIST",     2,  "binary-class",       False, "acc", 1, "breastmnist"),
    "BloodMNIST":     DatasetConfig("BloodMNIST",      8,  "multi-class",        False, "acc", 3, "bloodmnist"),
    "TissueMNIST":    DatasetConfig("TissueMNIST",     8,  "multi-class",        False, "acc", 1, "tissuemnist"),
    "RetinaMNIST":    DatasetConfig("RetinaMNIST",     5,  "ordinal-regression", False, "acc", 3, "retinamnist"),
    "OrganAMNIST":    DatasetConfig("OrganAMNIST",     11, "multi-class",        False, "acc", 1, "organamnist"),
    "OrganCMNIST":    DatasetConfig("OrganCMNIST",     11, "multi-class",        False, "acc", 1, "organcmnist"),
    "OrganSMNIST":    DatasetConfig("OrganSMNIST",     11, "multi-class",        False, "acc", 1, "organsmnist"),
    # --- 3D datasets (6) ---
    "OrganMNIST3D":   DatasetConfig("OrganMNIST3D",    11, "multi-class",        True,  "acc", 1, "organmnist3d"),
    "NoduleMNIST3D":  DatasetConfig("NoduleMNIST3D",   2,  "binary-class",       True,  "acc", 1, "nodulemnist3d"),
    "AdrenalMNIST3D": DatasetConfig("AdrenalMNIST3D",  2,  "binary-class",       True,  "acc", 1, "adrenalmnist3d"),
    "VesselMNIST3D":  DatasetConfig("VesselMNIST3D",   2,  "binary-class",       True,  "acc", 1, "vesselmnist3d"),
    "FractureMNIST3D":DatasetConfig("FractureMNIST3D", 3,  "multi-class",        True,  "acc", 1, "fracturemnist3d"),
    "SynapseMNIST3D": DatasetConfig("SynapseMNIST3D",  2,  "binary-class",       True,  "acc", 1, "synapsemnist3d"),
}


# ======================================================================
# Dataset lists
# ======================================================================

DATASETS_2D: List[str] = [
    "PathMNIST", "DermaMNIST", "OCTMNIST", "PneumoniaMNIST",
    "ChestMNIST", "BreastMNIST", "BloodMNIST", "TissueMNIST",
    "RetinaMNIST", "OrganAMNIST", "OrganCMNIST", "OrganSMNIST",
]

DATASETS_3D: List[str] = [
    "OrganMNIST3D", "NoduleMNIST3D", "AdrenalMNIST3D",
    "VesselMNIST3D", "FractureMNIST3D", "SynapseMNIST3D",
]

ALL_DATASETS: List[str] = DATASETS_2D + DATASETS_3D

INTERLEAVED_DATASETS: List[str] = [
    "PathMNIST",      "OrganMNIST3D",
    "DermaMNIST",     "NoduleMNIST3D",
    "OCTMNIST",       "AdrenalMNIST3D",
    "PneumoniaMNIST", "VesselMNIST3D",
    "ChestMNIST",     "FractureMNIST3D",
    "BreastMNIST",    "SynapseMNIST3D",
    "BloodMNIST", "TissueMNIST", "RetinaMNIST",
    "OrganAMNIST", "OrganCMNIST", "OrganSMNIST",
]


# ======================================================================
# Three-expert hard routing (MoE)
# ======================================================================

EXPERT_A = "A"   # Bio-Medical: RGB, microscopic texture
EXPERT_B = "B"   # Radiology:   grayscale, macro geometry
EXPERT_C = "C"   # Volumetric:  3D voxel, spatial structure

EXPERT_ROUTING: Dict[str, str] = {
    # Expert A — color is a key diagnostic feature
    "PathMNIST":   EXPERT_A,
    "BloodMNIST":  EXPERT_A,
    "TissueMNIST": EXPERT_A,
    "DermaMNIST":  EXPERT_A,
    "RetinaMNIST": EXPERT_A,
    # Expert B — shape / contour is a key diagnostic feature
    "ChestMNIST":     EXPERT_B,
    "PneumoniaMNIST": EXPERT_B,
    "BreastMNIST":    EXPERT_B,
    "OCTMNIST":       EXPERT_B,
    "OrganAMNIST":    EXPERT_B,
    "OrganCMNIST":    EXPERT_B,
    "OrganSMNIST":    EXPERT_B,
    # Expert C — volumetric spatial continuity
    "OrganMNIST3D":    EXPERT_C,
    "NoduleMNIST3D":   EXPERT_C,
    "AdrenalMNIST3D":  EXPERT_C,
    "VesselMNIST3D":   EXPERT_C,
    "FractureMNIST3D": EXPERT_C,
    "SynapseMNIST3D":  EXPERT_C,
}

EXPERT_BOTTLENECK: Dict[str, int] = {EXPERT_A: 64, EXPERT_B: 96, EXPERT_C: 128}

DATASETS_EXPERT_A = [k for k, v in EXPERT_ROUTING.items() if v == EXPERT_A]
DATASETS_EXPERT_B = [k for k, v in EXPERT_ROUTING.items() if v == EXPERT_B]
DATASETS_EXPERT_C = [k for k, v in EXPERT_ROUTING.items() if v == EXPERT_C]


# ======================================================================
# Utility helpers
# ======================================================================

def get_expert_id(dataset_name: str) -> str:
    return EXPERT_ROUTING[dataset_name]

def get_dataset_config(name: str) -> DatasetConfig:
    return DATASET_CONFIGS[name]

def get_num_classes_list(dataset_list: List[str]) -> List[int]:
    return [DATASET_CONFIGS[n].num_classes for n in dataset_list]

def get_task_id(dataset_name: str, dataset_list: List[str]) -> int:
    return dataset_list.index(dataset_name)

def is_3d_dataset(name: str) -> bool:
    return DATASET_CONFIGS[name].is_3d