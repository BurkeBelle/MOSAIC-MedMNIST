# model/lora_adapter.py
"""
LoRA (Low-Rank Adaptation) for ViT.

Reference: LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)

Design:
    1. Add low-rank decomposition to Attention Q and V projections: h = Wx + BAx
    2. A initialized with Kaiming, B zeroed -> LoRA output is 0 at init
    3. scaling = alpha / rank
    4. Drop-in compatible with TransformerBlockWithAdapter interface
"""

import math
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Single LoRA low-rank decomposition layer.

    h = Wx + (B @ A)(x) * scaling

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        rank:         Low-rank dimension (default 8).
        alpha:        Scaling coefficient (default 16).
        dropout:      Dropout probability.
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """[B, N, in] -> [B, N, out] (LoRA delta only)."""
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAAttention(nn.Module):
    """
    Multi-Head Self Attention with LoRA on Q and V projections.

    Args:
        dim / num_heads / qkv_bias / attn_drop / proj_drop: Standard ViT params.
        lora_rank / lora_alpha / lora_dropout: LoRA params.
    """

    def __init__(self, dim=768, num_heads=12, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0,
                 lora_rank=8, lora_alpha=16.0, lora_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim

        # Original projections (frozen)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # LoRA on Q and V (trainable)
        self.lora_q = LoRALayer(dim, dim, lora_rank, lora_alpha, lora_dropout)
        self.lora_v = LoRALayer(dim, dim, lora_rank, lora_alpha, lora_dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Add LoRA deltas
        q = q + self.lora_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v + self.lora_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class LoRATransformerBlock(nn.Module):
    """
    Transformer block with LoRA inside Attention.

    Same structure as vanilla ViT block, but Q and V have LoRA branches.
    Compatible with TransformerBlockWithAdapter interface (is_3d / expert_id ignored).
    """

    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0,
                 lora_rank=8, lora_alpha=16.0, lora_dropout=0.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = LoRAAttention(dim, num_heads, qkv_bias, attn_drop, drop,
                                  lora_rank, lora_alpha, lora_dropout)
        from .transformer_block import FFN
        self.ffn = FFN(dim, int(dim * mlp_ratio), drop)
        self.use_adapter = True
        self.adapter_mode = "lora"

    def forward(self, x, is_3d=False, expert_id=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LoRATransformerEncoder(nn.Module):
    """Multi-layer Transformer encoder with LoRA. Drop-in for TransformerEncoder."""

    def __init__(self, depth=12, dim=768, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0,
                 lora_rank=8, lora_alpha=16.0, lora_dropout=0.0, **kwargs):
        super().__init__()
        self.depth = depth
        self.use_adapter = True
        self.adapter_mode = "lora"

        self.blocks = nn.ModuleList([
            LoRATransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                                lora_rank, lora_alpha, lora_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        lp = sum(p.numel() for n, p in self.named_parameters() if "lora" in n)
        print(f"LoRATransformerEncoder:")
        print(f"  Depth: {depth}, Rank: {lora_rank}, Alpha: {lora_alpha}")
        print(f"  LoRA params per layer: {lp // depth:,}")
        print(f"  Total LoRA params: {lp:,} ({lp/1e6:.2f}M)")

    def forward(self, x, is_3d=False, expert_id=None):
        for blk in self.blocks:
            x = blk(x, is_3d=is_3d, expert_id=expert_id)
        return self.norm(x)