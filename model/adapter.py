# model/adapter.py
"""
AdaptFormer Adapter Modules

Reference: AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition (NeurIPS 2022)

Architecture:
    1. Bottleneck structure: Down -> ReLU -> Up
    2. Residual connection + Scale factor
    3. LoRA-style init: up_proj zeroed so adapter output is 0 at initialization
"""

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """
    Single adapter module (AdaptFormer style).

    Structure: Input -> LayerNorm -> Down_proj -> ReLU -> Up_proj -> Scale -> Output

    Args:
        d_model:        Input dimension (ViT-Base: 768).
        bottleneck:     Bottleneck dimension (default 64, recommended by paper).
        dropout:        Dropout probability.
        init_option:    Weight init method ("lora" or "bert").
        adapter_scalar: Scale factor (paper recommends 0.1).
    """

    def __init__(
        self,
        d_model: int = 768,
        bottleneck: int = 64,
        dropout: float = 0.0,
        init_option: str = "lora",
        adapter_scalar: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck

        self.layer_norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = adapter_scalar

        self._init_weights(init_option)

    def _init_weights(self, init_option: str):
        if init_option == "lora":
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)
        elif init_option == "bert":
            nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
        else:
            raise ValueError(f"Unknown init_option: {init_option}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, D].
        Returns:
            Adapter output [B, N, D] (residual handled by caller).
        """
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x * self.scale


class ModalityAdapter(nn.Module):
    """
    Modality-specific adapter (V1: 2D / 3D dual-channel).

    Contains separate 2D and 3D adapters; routes by input modality.

    Args:
        d_model:        Input dimension.
        bottleneck:     Bottleneck dimension.
        dropout:        Dropout probability.
        init_option:    Weight init method.
        adapter_scalar: Scale factor.
    """

    def __init__(
        self,
        d_model: int = 768,
        bottleneck: int = 64,
        dropout: float = 0.0,
        init_option: str = "lora",
        adapter_scalar: float = 0.1,
    ):
        super().__init__()
        self.adapter_2d = Adapter(d_model, bottleneck, dropout, init_option, adapter_scalar)
        self.adapter_3d = Adapter(d_model, bottleneck, dropout, init_option, adapter_scalar)

    def forward(self, x: torch.Tensor, is_3d: bool = False) -> torch.Tensor:
        return self.adapter_3d(x) if is_3d else self.adapter_2d(x)


class MoEAdapter(nn.Module):
    """
    Three-expert Mixture-of-Specialists adapter (V2: hard routing).

    Experts:
        A (Bio-Medical): RGB images, microscopic texture   -> bottleneck_a
        B (Radiology):   Grayscale, macro geometry          -> bottleneck_b
        C (Volumetric):  3D voxel, spatial structure        -> bottleneck_c

    Routing is hard-coded by dataset name (no learned gating).

    Args:
        d_model:      Input dimension (ViT-Base: 768).
        bottleneck_a: Expert A bottleneck dim (default 64).
        bottleneck_b: Expert B bottleneck dim (default 96).
        bottleneck_c: Expert C bottleneck dim (default 128).
        dropout:      Dropout probability.
        init_option:  Weight init method.
        adapter_scalar: Scale factor.
    """

    def __init__(
        self,
        d_model: int = 768,
        bottleneck_a: int = 64,
        bottleneck_b: int = 96,
        bottleneck_c: int = 128,
        dropout: float = 0.0,
        init_option: str = "lora",
        adapter_scalar: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_a = bottleneck_a
        self.bottleneck_b = bottleneck_b
        self.bottleneck_c = bottleneck_c

        self.expert_a = Adapter(d_model, bottleneck_a, dropout, init_option, adapter_scalar)
        self.expert_b = Adapter(d_model, bottleneck_b, dropout, init_option, adapter_scalar)
        self.expert_c = Adapter(d_model, bottleneck_c, dropout, init_option, adapter_scalar)

    def forward(self, x: torch.Tensor, expert_id: str = "A") -> torch.Tensor:
        """
        Hard-routed forward pass.

        Args:
            x:          Input features [B, N, D].
            expert_id:  Expert selector ('A', 'B', or 'C').
        """
        if expert_id == "A":
            return self.expert_a(x)
        elif expert_id == "B":
            return self.expert_b(x)
        elif expert_id == "C":
            return self.expert_c(x)
        raise ValueError(f"Unknown expert_id: {expert_id}. Must be 'A', 'B', or 'C'")

    def get_expert(self, expert_id: str) -> Adapter:
        return {"A": self.expert_a, "B": self.expert_b, "C": self.expert_c}[expert_id]

    def get_expert_params(self, expert_id: str):
        return self.get_expert(expert_id).parameters()