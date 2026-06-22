# model/unified_model.py
"""
MOSAIC Unified 2D / 3D Medical Image Classification Model.

Architecture:
    1. MedCoSS-style tokenizer  — 2D/3D -> [B, N, D]
    2. Transformer backbone     — shared attention layers (frozen)
    3. AdaptFormer adapters     — modality-specific, parallel to FFN
    4. Multi-task heads          — one linear head per dataset

Supports:
    V1 (ModalityAdapter): 2D / 3D dual-channel adapters.
    V2 (MoEAdapter):      Three hard-routed experts (A / B / C).
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List

from .patch_embed import UnifiedPatchEmbed
from .transformer_block import TransformerEncoder


class UnifiedModel(nn.Module):
    """
    Unified 2D / 3D medical image classification model.

    Args:
        num_classes_list:   Number of classes per task.
        embed_dim:          Embedding dimension (default 768).
        depth:              Transformer depth (default 12).
        num_heads:          Attention heads (default 12).
        use_adapter:        Attach adapters to transformer blocks.
        adapter_mode:       'v1' (ModalityAdapter) or 'v2_moe' (MoEAdapter).
        adapter_bottleneck: V1 bottleneck dim.
        adapter_bottleneck_a/b/c: V2 per-expert bottleneck dims.
        adapter_scalar:     Adapter output scaling factor.
    """

    def __init__(
        self,
        num_classes_list: List[int],
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_adapter: bool = True,
        adapter_mode: str = "v1",
        adapter_bottleneck: int = 64,
        adapter_bottleneck_a: int = 64,
        adapter_bottleneck_b: int = 96,
        adapter_bottleneck_c: int = 128,
        adapter_dropout: float = 0.0,
        adapter_scalar: float = 0.1,
        img_size_2d: int = 224,
        patch_size_2d: int = 16,
        in_chans_2d: int = 3,
        img_size_3d: int = 64,
        patch_size_3d: int = 8,
        in_chans_3d: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_adapter = use_adapter
        self.adapter_mode = adapter_mode
        self.num_tasks = len(num_classes_list)

        # 1. Patch embedding (tokenizer)
        self.patch_embed = UnifiedPatchEmbed(
            img_size_2d, patch_size_2d, in_chans_2d,
            img_size_3d, patch_size_3d, in_chans_3d,
            embed_dim,
        )

        # 2. CLS token & position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_2d = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches_2d + 1, embed_dim)
        )
        self.pos_embed_3d = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches_3d + 1, embed_dim)
        )

        # 3. Transformer encoder with adapters
        self.encoder = TransformerEncoder(
            depth=depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop_rate, attn_drop=attn_drop_rate,
            use_adapter=use_adapter, adapter_mode=adapter_mode,
            adapter_bottleneck=adapter_bottleneck,
            adapter_bottleneck_a=adapter_bottleneck_a,
            adapter_bottleneck_b=adapter_bottleneck_b,
            adapter_bottleneck_c=adapter_bottleneck_c,
            adapter_dropout=adapter_dropout, adapter_scalar=adapter_scalar,
        )

        # 4. Multi-task classification heads
        self.heads = nn.ModuleList(
            [nn.Linear(embed_dim, nc) for nc in num_classes_list]
        )

        # Init
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_3d, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for head in self.heads:
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    # ------------------------------------------------------------------
    def forward_features(self, x, expert_id=None):
        tokens, is_3d = self.patch_embed(x)
        B = tokens.shape[0]
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        tokens = tokens + (self.pos_embed_3d if is_3d else self.pos_embed_2d)
        tokens = self.encoder(tokens, is_3d=is_3d, expert_id=expert_id)
        return tokens[:, 0], is_3d

    def forward(self, x, task_id=0, return_features=False, expert_id=None):
        features, _ = self.forward_features(x, expert_id)
        if return_features:
            return features
        return features, self.heads[task_id](features)

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze everything except adapters and classification heads."""
        for name, param in self.named_parameters():
            if "adapter" not in name and "heads" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


class TeacherModel(nn.Module):
    """
    EMA teacher model.

    Mirrors the student architecture; updated via exponential moving average.
    Supports modality-aware (V1) and expert-aware (V2) selective EMA updates
    to prevent cross-modality parameter contamination.
    """

    def __init__(self, student: UnifiedModel):
        super().__init__()
        self.model = copy.deepcopy(student)
        self.adapter_mode = student.adapter_mode
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, task_id=0, return_features=False, expert_id=None):
        return self.model(x, task_id=task_id, return_features=return_features,
                          expert_id=expert_id)

    @torch.no_grad()
    def ema_update(self, student, momentum=0.999, is_3d=None, expert_id=None):
        for (name_t, p_t), (_, p_s) in zip(
            self.model.named_parameters(), student.named_parameters()
        ):
            # V2 MoE: only update the active expert
            if self.adapter_mode == "v2_moe" and expert_id is not None:
                if "expert_a" in name_t and expert_id != "A":
                    continue
                if "expert_b" in name_t and expert_id != "B":
                    continue
                if "expert_c" in name_t and expert_id != "C":
                    continue
            # V1: only update the active modality adapter
            elif self.adapter_mode == "v1" and is_3d is not None:
                if "adapter_2d" in name_t and is_3d:
                    continue
                if "adapter_3d" in name_t and not is_3d:
                    continue
            p_t.data.mul_(momentum).add_((1 - momentum) * p_s.data)


# ======================================================================
# Weight loading
# ======================================================================

def load_pretrained_vit_from_npz(model: UnifiedModel, npz_path: str):
    """Load ViT-Base weights from a JAX/Flax .npz checkpoint into *model*."""
    print(f"Loading ViT pretrained weights from: {npz_path}")
    weights = np.load(npz_path)
    loaded = 0

    with torch.no_grad():
        # Patch embedding (2D only)
        if "embedding/kernel" in weights:
            kernel = np.transpose(weights["embedding/kernel"], (3, 2, 0, 1))
            model.patch_embed.patch_embed_2d.proj.weight.data = torch.from_numpy(kernel).float()
            loaded += 1
        if "embedding/bias" in weights:
            model.patch_embed.patch_embed_2d.proj.bias.data = (
                torch.from_numpy(weights["embedding/bias"]).float()
            )
            loaded += 1

        # CLS token
        if "cls" in weights:
            model.cls_token.data = torch.from_numpy(weights["cls"]).float()
            loaded += 1

        # Position embedding (2D)
        key_pos = "Transformer/posembed_input/pos_embedding"
        if key_pos in weights:
            pos = weights[key_pos]
            if pos.shape[1] == model.pos_embed_2d.shape[1]:
                model.pos_embed_2d.data = torch.from_numpy(pos).float()
                loaded += 1

        # Transformer blocks
        for i in range(model.encoder.depth):
            pfx = f"Transformer/encoderblock_{i}"
            blk = model.encoder.blocks[i]

            # LayerNorm 1
            if f"{pfx}/LayerNorm_0/scale" in weights:
                blk.norm1.weight.data = torch.from_numpy(weights[f"{pfx}/LayerNorm_0/scale"]).float()
                blk.norm1.bias.data = torch.from_numpy(weights[f"{pfx}/LayerNorm_0/bias"]).float()
                loaded += 2

            # QKV
            qk = f"{pfx}/MultiHeadDotProductAttention_1"
            if f"{qk}/query/kernel" in weights:
                q_w = weights[f"{qk}/query/kernel"].reshape(768, -1).T
                k_w = weights[f"{qk}/key/kernel"].reshape(768, -1).T
                v_w = weights[f"{qk}/value/kernel"].reshape(768, -1).T
                blk.attn.qkv.weight.data = torch.from_numpy(np.concatenate([q_w, k_w, v_w])).float()
                q_b = weights[f"{qk}/query/bias"].reshape(-1)
                k_b = weights[f"{qk}/key/bias"].reshape(-1)
                v_b = weights[f"{qk}/value/bias"].reshape(-1)
                blk.attn.qkv.bias.data = torch.from_numpy(np.concatenate([q_b, k_b, v_b])).float()
                loaded += 2

            # Output projection
            if f"{qk}/out/kernel" in weights:
                blk.attn.proj.weight.data = torch.from_numpy(
                    weights[f"{qk}/out/kernel"].reshape(-1, 768).T
                ).float()
                blk.attn.proj.bias.data = torch.from_numpy(weights[f"{qk}/out/bias"]).float()
                loaded += 2

            # LayerNorm 2
            if f"{pfx}/LayerNorm_2/scale" in weights:
                blk.norm2.weight.data = torch.from_numpy(weights[f"{pfx}/LayerNorm_2/scale"]).float()
                blk.norm2.bias.data = torch.from_numpy(weights[f"{pfx}/LayerNorm_2/bias"]).float()
                loaded += 2

            # FFN
            mlp = f"{pfx}/MlpBlock_3"
            if f"{mlp}/Dense_0/kernel" in weights:
                blk.ffn.fc1.weight.data = torch.from_numpy(weights[f"{mlp}/Dense_0/kernel"].T).float()
                blk.ffn.fc1.bias.data = torch.from_numpy(weights[f"{mlp}/Dense_0/bias"]).float()
                blk.ffn.fc2.weight.data = torch.from_numpy(weights[f"{mlp}/Dense_1/kernel"].T).float()
                blk.ffn.fc2.bias.data = torch.from_numpy(weights[f"{mlp}/Dense_1/bias"]).float()
                loaded += 4

        # Final LayerNorm
        if "Transformer/encoder_norm/scale" in weights:
            model.encoder.norm.weight.data = torch.from_numpy(
                weights["Transformer/encoder_norm/scale"]
            ).float()
            model.encoder.norm.bias.data = torch.from_numpy(
                weights["Transformer/encoder_norm/bias"]
            ).float()
            loaded += 2

    print(f"  Loaded {loaded} weight tensors")
    print(f"  Note: 3D patch embed, adapters, and heads are randomly initialized")


# ======================================================================
# Factory
# ======================================================================

def create_model_and_teacher(
    num_classes_list,
    use_adapter=True,
    adapter_mode="v1",
    adapter_bottleneck=64,
    adapter_bottleneck_a=64,
    adapter_bottleneck_b=96,
    adapter_bottleneck_c=128,
    adapter_scalar=0.1,
    pretrained_path=None,
):
    """Create a student–teacher pair, optionally loading pretrained ViT weights."""
    student = UnifiedModel(
        num_classes_list=num_classes_list,
        use_adapter=use_adapter,
        adapter_mode=adapter_mode,
        adapter_bottleneck=adapter_bottleneck,
        adapter_bottleneck_a=adapter_bottleneck_a,
        adapter_bottleneck_b=adapter_bottleneck_b,
        adapter_bottleneck_c=adapter_bottleneck_c,
        adapter_scalar=adapter_scalar,
    )

    if pretrained_path is not None and pretrained_path.endswith(".npz"):
        load_pretrained_vit_from_npz(student, pretrained_path)

    teacher = TeacherModel(student)
    return student, teacher