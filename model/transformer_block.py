# model/transformer_block.py
"""
Transformer Block with Adapter (AdaptFormer style).

Structure:
    x -> LN -> MHSA -> + -> LN -> FFN     -> +
                                   Adapter -> |
                                              (parallel sum)

Supports two adapter modes:
    V1 (ModalityAdapter): 2D / 3D dual-channel.
    V2 (MoEAdapter):      Three hard-routed experts (Bio-Medical / Radiology / Volumetric).
"""

import torch
import torch.nn as nn
from typing import Optional

from .adapter import ModalityAdapter, MoEAdapter


class Attention(nn.Module):
    """Standard Multi-Head Self Attention (ViT style)."""

    def __init__(self, dim=768, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class FFN(nn.Module):
    """Feed-Forward Network: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, dim=768, hidden_dim=3072, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlockWithAdapter(nn.Module):
    """
    Transformer block with a parallel adapter branch.

    Forward:
        x'  = x  + MHSA(LN(x))
        x'' = x' + FFN(LN(x')) + Adapter(LN(x'))

    Args:
        dim:            Feature dimension.
        num_heads:      Number of attention heads.
        mlp_ratio:      FFN hidden dimension multiplier.
        use_adapter:    Whether to attach an adapter.
        adapter_mode:   'v1' (ModalityAdapter) or 'v2_moe' (MoEAdapter).
        adapter_bottleneck:     V1 bottleneck dim.
        adapter_bottleneck_a/b/c: V2 per-expert bottleneck dims.
        adapter_scalar: Adapter output scaling factor.
    """

    def __init__(
        self,
        dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
        drop=0.0, attn_drop=0.0,
        use_adapter=True, adapter_mode="v1",
        adapter_bottleneck=64,
        adapter_bottleneck_a=64, adapter_bottleneck_b=96, adapter_bottleneck_c=128,
        adapter_dropout=0.0, adapter_scalar=0.1,
    ):
        super().__init__()
        self.use_adapter = use_adapter
        self.adapter_mode = adapter_mode

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.ffn = FFN(dim, int(dim * mlp_ratio), drop)

        if use_adapter:
            if adapter_mode == "v1":
                self.adapter = ModalityAdapter(
                    dim, adapter_bottleneck, adapter_dropout, "lora", adapter_scalar
                )
            elif adapter_mode == "v2_moe":
                self.adapter = MoEAdapter(
                    dim, adapter_bottleneck_a, adapter_bottleneck_b,
                    adapter_bottleneck_c, adapter_dropout, "lora", adapter_scalar
                )
            else:
                raise ValueError(f"Unknown adapter_mode: {adapter_mode}")

    def forward(self, x, is_3d=False, expert_id=None):
        x = x + self.attn(self.norm1(x))
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)

        if self.use_adapter:
            if self.adapter_mode == "v1":
                adapter_out = self.adapter(normed, is_3d=is_3d)
            else:
                adapter_out = self.adapter(normed, expert_id=expert_id)
            return x + ffn_out + adapter_out

        return x + ffn_out


class TransformerEncoder(nn.Module):
    """
    Multi-layer Transformer encoder with optional adapters.

    Args:
        depth:      Number of layers (ViT-Base: 12).
        dim:        Feature dimension.
        num_heads:  Number of attention heads.
        All adapter-related args forwarded to TransformerBlockWithAdapter.
    """

    def __init__(
        self,
        depth=12, dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
        drop=0.0, attn_drop=0.0,
        use_adapter=True, adapter_mode="v1",
        adapter_bottleneck=64,
        adapter_bottleneck_a=64, adapter_bottleneck_b=96, adapter_bottleneck_c=128,
        adapter_dropout=0.0, adapter_scalar=0.1,
    ):
        super().__init__()
        self.depth = depth
        self.use_adapter = use_adapter
        self.adapter_mode = adapter_mode

        self.blocks = nn.ModuleList([
            TransformerBlockWithAdapter(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                use_adapter=use_adapter, adapter_mode=adapter_mode,
                adapter_bottleneck=adapter_bottleneck,
                adapter_bottleneck_a=adapter_bottleneck_a,
                adapter_bottleneck_b=adapter_bottleneck_b,
                adapter_bottleneck_c=adapter_bottleneck_c,
                adapter_dropout=adapter_dropout, adapter_scalar=adapter_scalar,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, is_3d=False, expert_id=None):
        for block in self.blocks:
            x = block(x, is_3d=is_3d, expert_id=expert_id)
        return self.norm(x)