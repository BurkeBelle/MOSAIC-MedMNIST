# model/__init__.py
"""
Model modules.
- Adapter: AdaptFormer-style adapters (V1: ModalityAdapter, V2: MoEAdapter)
- PatchEmbed: Unified 2D/3D Patch Embedding
- TransformerBlock: Transformer layers with parallel adapters
- UnifiedModel: MOSAIC unified 2D/3D classification model
- TeacherModel: EMA teacher model
- Baseline: LoRA / VPT PEFT baselines
"""

# ============================================================
# MOSAIC core modules
# ============================================================
from .adapter import Adapter, ModalityAdapter, MoEAdapter
from .patch_embed import PatchEmbed2D, PatchEmbed3D, UnifiedPatchEmbed
from .transformer_block import (
    Attention,
    FFN,
    TransformerBlockWithAdapter,
    TransformerEncoder,
)
from .unified_model import (
    UnifiedModel,
    TeacherModel,
    create_model_and_teacher,
)

# ============================================================
# PEFT baseline modules
# ============================================================
from .lora_adapter import LoRALayer, LoRAAttention, LoRATransformerEncoder
from .vpt_adapter import VPTTransformerEncoder
from .baseline_model import BaselineModel, BaselineTeacher, create_baseline_model

__all__ = [
    # Adapter
    "Adapter", "ModalityAdapter", "MoEAdapter",
    # Patch Embedding
    "PatchEmbed2D", "PatchEmbed3D", "UnifiedPatchEmbed",
    # Transformer
    "Attention", "FFN", "TransformerBlockWithAdapter", "TransformerEncoder",
    # MOSAIC Model
    "UnifiedModel", "TeacherModel", "create_model_and_teacher",
    # Baseline PEFT
    "LoRALayer", "LoRAAttention", "LoRATransformerEncoder",
    "VPTTransformerEncoder",
    "BaselineModel", "BaselineTeacher", "create_baseline_model",
]