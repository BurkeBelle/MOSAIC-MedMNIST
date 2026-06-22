# model/vpt_adapter.py
"""
VPT (Visual Prompt Tuning) for ViT.

Reference: Visual Prompt Tuning (ECCV 2022)

Implements VPT-Deep:
    1. Prepend learnable prompt tokens at every Transformer layer.
    2. Prompts participate in self-attention but are removed from output.
    3. Only prompt parameters are trainable.

Each layer has its own independent prompt tokens (VPT-Deep).
Interface is compatible with TransformerBlockWithAdapter.
"""

import torch
import torch.nn as nn


class VPTTransformerBlock(nn.Module):
    """
    Transformer block with VPT-Deep prompts.

    Forward:
        x [B, N, D]
        -> prepend prompts: [B, P+N, D]
        -> LN -> MHSA -> residual
        -> LN -> FFN  -> residual
        -> remove prompts: [B, N, D]

    Args:
        dim / num_heads / mlp_ratio: Standard ViT params.
        num_prompts:    Number of prompt tokens per layer.
        prompt_dropout: Dropout on prompt tokens.
    """

    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0,
                 num_prompts=10, prompt_dropout=0.0, **kwargs):
        super().__init__()
        self.num_prompts = num_prompts
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        from .transformer_block import Attention, FFN
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.ffn = FFN(dim, int(dim * mlp_ratio), drop)

        # Learnable prompt tokens
        self.prompt_tokens = nn.Parameter(torch.zeros(1, num_prompts, dim))
        nn.init.trunc_normal_(self.prompt_tokens, std=0.02)
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        self.use_adapter = True
        self.adapter_mode = "vpt"

    def forward(self, x, is_3d=False, expert_id=None):
        B = x.shape[0]
        prompts = self.prompt_dropout(self.prompt_tokens.expand(B, -1, -1))
        xp = torch.cat([prompts, x], dim=1)         # [B, P+N, D]
        xp = xp + self.attn(self.norm1(xp))
        xp = xp + self.ffn(self.norm2(xp))
        return xp[:, self.num_prompts:, :]           # [B, N, D]


class VPTTransformerEncoder(nn.Module):
    """Multi-layer Transformer encoder with VPT-Deep. Drop-in for TransformerEncoder."""

    def __init__(self, depth=12, dim=768, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0,
                 num_prompts=10, prompt_dropout=0.0, **kwargs):
        super().__init__()
        self.depth = depth
        self.use_adapter = True
        self.adapter_mode = "vpt"

        self.blocks = nn.ModuleList([
            VPTTransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                                num_prompts, prompt_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        vp = sum(p.numel() for n, p in self.named_parameters() if "prompt" in n)
        print(f"VPTTransformerEncoder (VPT-Deep):")
        print(f"  Depth: {depth}, Prompts per layer: {num_prompts}")
        print(f"  VPT params per layer: {num_prompts * dim:,}")
        print(f"  Total VPT params: {vp:,} ({vp/1e6:.2f}M)")

    def forward(self, x, is_3d=False, expert_id=None):
        for blk in self.blocks:
            x = blk(x, is_3d=is_3d, expert_id=expert_id)
        return self.norm(x)