# model/patch_embed.py
"""
2D / 3D Patch Embedding (MedCoSS style).

Converts raw images into token sequences for the Transformer backbone:
    2D: [B, C, H, W]       -> [B, N, D]   where N = (H*W) / P^2
    3D: [B, C, D, H, W]    -> [B, N, D]   where N = (D*H*W) / P^3

Unified output dimension D allows 2D and 3D data to share the same backbone.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbed2D(nn.Module):
    """
    2D image patch embedding.

    Args:
        img_size:   Input image size (default 224).
        patch_size: Patch size (default 16).
        in_chans:   Input channels (RGB = 3).
        embed_dim:  Embedding dimension (ViT-Base = 768).
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, D, H', W'] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class PatchEmbed3D(nn.Module):
    """
    3D volume patch embedding.

    Args:
        img_size:   Input volume size (default 64, i.e. 64x64x64).
        patch_size: Patch size (default 8, i.e. 8x8x8).
        in_chans:   Input channels (typically 1).
        embed_dim:  Embedding dimension (ViT-Base = 768).
    """

    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 3
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, D, H, W] -> [B, embed_dim, D', H', W'] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class UnifiedPatchEmbed(nn.Module):
    """
    Unified 2D / 3D patch embedding.

    Automatically dispatches to 2D or 3D based on input tensor dimensionality.

    Args:
        img_size_2d / patch_size_2d / in_chans_2d: 2D settings.
        img_size_3d / patch_size_3d / in_chans_3d: 3D settings.
        embed_dim: Shared embedding dimension.
    """

    def __init__(
        self,
        img_size_2d=224, patch_size_2d=16, in_chans_2d=3,
        img_size_3d=64,  patch_size_3d=8,  in_chans_3d=1,
        embed_dim=768,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed_2d = PatchEmbed2D(img_size_2d, patch_size_2d, in_chans_2d, embed_dim)
        self.patch_embed_3d = PatchEmbed3D(img_size_3d, patch_size_3d, in_chans_3d, embed_dim)
        self.num_patches_2d = self.patch_embed_2d.num_patches
        self.num_patches_3d = self.patch_embed_3d.num_patches

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Returns:
            tokens: [B, N, D]
            is_3d:  Whether the input is 3D.
        """
        if x.dim() == 4:
            return self.patch_embed_2d(x), False
        elif x.dim() == 5:
            return self.patch_embed_3d(x), True
        raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")