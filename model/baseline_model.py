# model/baseline_model.py
"""
Baseline model factory.

Supports four PEFT configurations:
    adapter_mode='lora'    -> LoRA on Q, V (standard LoRA)
    adapter_mode='vpt'     -> VPT-Deep (learnable prompts per layer)
    adapter_mode='v2_moe'  -> MOSAIC MoE Adapter (original)
    adapter_mode='none'    -> No adapter (linear probe baseline)

All configurations share:
    - Frozen ViT-Base backbone (ImageNet pretrained .npz)
    - MedCoSS-style 2D/3D Tokenizer (UnifiedPatchEmbed)
    - Multi-task classification heads
"""

import copy
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .patch_embed import UnifiedPatchEmbed
from .transformer_block import TransformerEncoder
from .lora_adapter import LoRATransformerEncoder
from .vpt_adapter import VPTTransformerEncoder


class BaselineModel(nn.Module):
    """
    Unified baseline model.

    Selects the PEFT strategy via adapter_mode while keeping the tokenizer,
    backbone weights, and classification heads identical across methods.

    Args:
        num_classes_list: Number of classes per task.
        adapter_mode:     'lora' | 'vpt' | 'v2_moe' | 'v1' | 'none'.
        lora_rank / lora_alpha:       LoRA hyper-params.
        num_prompts:                  VPT number of prompt tokens.
        adapter_bottleneck_a/b/c:     MoE per-expert bottleneck dims.
    """

    def __init__(
        self,
        num_classes_list: List[int],
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0,
        # PEFT selection
        adapter_mode="lora",
        # LoRA
        lora_rank=8, lora_alpha=16.0, lora_dropout=0.0,
        # VPT
        num_prompts=10, prompt_dropout=0.0,
        # MoE (original MOSAIC)
        adapter_bottleneck=64,
        adapter_bottleneck_a=64, adapter_bottleneck_b=96, adapter_bottleneck_c=128,
        adapter_dropout=0.0, adapter_scalar=0.1,
        # Tokenizer
        img_size_2d=224, patch_size_2d=16, in_chans_2d=3,
        img_size_3d=64,  patch_size_3d=8,  in_chans_3d=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_mode = adapter_mode
        self.use_adapter = (adapter_mode != "none")
        self.num_tasks = len(num_classes_list)

        # 1. Patch embedding (shared, identical to MOSAIC)
        self.patch_embed = UnifiedPatchEmbed(
            img_size_2d, patch_size_2d, in_chans_2d,
            img_size_3d, patch_size_3d, in_chans_3d, embed_dim,
        )

        # 2. CLS token + position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_2d = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches_2d + 1, embed_dim))
        self.pos_embed_3d = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches_3d + 1, embed_dim))

        # 3. Encoder (selected by adapter_mode)
        if adapter_mode == "lora":
            self.encoder = LoRATransformerEncoder(
                depth=depth, dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        elif adapter_mode == "vpt":
            self.encoder = VPTTransformerEncoder(
                depth=depth, dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                num_prompts=num_prompts, prompt_dropout=prompt_dropout)
        elif adapter_mode in ("v1", "v2_moe"):
            self.encoder = TransformerEncoder(
                depth=depth, dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                use_adapter=True, adapter_mode=adapter_mode,
                adapter_bottleneck=adapter_bottleneck,
                adapter_bottleneck_a=adapter_bottleneck_a,
                adapter_bottleneck_b=adapter_bottleneck_b,
                adapter_bottleneck_c=adapter_bottleneck_c,
                adapter_dropout=adapter_dropout, adapter_scalar=adapter_scalar)
        elif adapter_mode == "none":
            self.encoder = TransformerEncoder(
                depth=depth, dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                use_adapter=False)
        else:
            raise ValueError(f"Unknown adapter_mode: {adapter_mode}")

        # 4. Multi-task classification heads
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nc) for nc in num_classes_list])

        # Init
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_3d, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for h in self.heads:
            nn.init.trunc_normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

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

    def freeze_backbone(self):
        """Freeze everything, then unfreeze PEFT params + heads."""
        for p in self.parameters():
            p.requires_grad = False
        for h in self.heads:
            for p in h.parameters():
                p.requires_grad = True

        patterns = {"lora": "lora", "vpt": "prompt",
                    "v1": "adapter", "v2_moe": "adapter"}
        pat = patterns.get(self.adapter_mode)
        if pat:
            for n, p in self.named_parameters():
                if pat in n or "expert" in n:
                    p.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nfreeze_backbone ({self.adapter_mode}):")
        print(f"  Total: {total:,} ({total/1e6:.2f}M)")
        print(f"  Trainable: {train:,} ({train/1e6:.2f}M)")
        print(f"  Trainable ratio: {train/total*100:.2f}%")


class BaselineTeacher(nn.Module):
    """
    EMA teacher model for all PEFT modes.

    EMA strategy:
        LoRA / VPT Joint: update all parameters (no expert isolation).
        MoE:              expert-aware selective update (original MOSAIC).
    """

    def __init__(self, student: BaselineModel):
        super().__init__()
        self.model = copy.deepcopy(student)
        self.adapter_mode = student.adapter_mode
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, task_id=0, return_features=False, expert_id=None):
        return self.model(x, task_id=task_id,
                          return_features=return_features, expert_id=expert_id)

    @torch.no_grad()
    def ema_update(self, student, momentum=0.999, is_3d=None, expert_id=None):
        for (nt, pt), (_, ps) in zip(
            self.model.named_parameters(), student.named_parameters()
        ):
            if self.adapter_mode == "v2_moe" and expert_id is not None:
                if "expert_a" in nt and expert_id != "A": continue
                if "expert_b" in nt and expert_id != "B": continue
                if "expert_c" in nt and expert_id != "C": continue
            elif self.adapter_mode == "v1" and is_3d is not None:
                if "adapter_2d" in nt and is_3d: continue
                if "adapter_3d" in nt and not is_3d: continue
            pt.data.mul_(momentum).add_((1 - momentum) * ps.data)


# ======================================================================
# Factory & weight loading
# ======================================================================

def create_baseline_model(num_classes_list, adapter_mode="lora",
                          pretrained_path=None, freeze_backbone=True,
                          **kwargs):
    """Create student + teacher pair."""
    student = BaselineModel(num_classes_list=num_classes_list,
                            adapter_mode=adapter_mode, **kwargs)
    if pretrained_path and pretrained_path.endswith(".npz"):
        _load_npz_weights(student, pretrained_path)
    if freeze_backbone:
        student.freeze_backbone()
    return student, BaselineTeacher(student)


def _load_npz_weights(model, npz_path):
    """Load ViT-Base JAX/Flax .npz weights (compatible with all adapter modes)."""
    import numpy as np
    print(f"Loading ViT pretrained weights from: {npz_path}")
    w = np.load(npz_path)
    n = 0

    with torch.no_grad():
        if "embedding/kernel" in w:
            model.patch_embed.patch_embed_2d.proj.weight.data = torch.from_numpy(
                np.transpose(w["embedding/kernel"], (3,2,0,1))).float(); n += 1
        if "embedding/bias" in w:
            model.patch_embed.patch_embed_2d.proj.bias.data = torch.from_numpy(
                w["embedding/bias"]).float(); n += 1
        if "cls" in w:
            model.cls_token.data = torch.from_numpy(w["cls"]).float(); n += 1
        pk = "Transformer/posembed_input/pos_embedding"
        if pk in w and w[pk].shape[1] == model.pos_embed_2d.shape[1]:
            model.pos_embed_2d.data = torch.from_numpy(w[pk]).float(); n += 1

        for i in range(model.encoder.depth):
            p = f"Transformer/encoderblock_{i}"
            b = model.encoder.blocks[i]
            if f"{p}/LayerNorm_0/scale" in w:
                b.norm1.weight.data = torch.from_numpy(w[f"{p}/LayerNorm_0/scale"]).float()
                b.norm1.bias.data = torch.from_numpy(w[f"{p}/LayerNorm_0/bias"]).float(); n += 2
            ak = f"{p}/MultiHeadDotProductAttention_1"
            if f"{ak}/query/kernel" in w:
                qw = w[f"{ak}/query/kernel"].reshape(768,-1).T
                kw = w[f"{ak}/key/kernel"].reshape(768,-1).T
                vw = w[f"{ak}/value/kernel"].reshape(768,-1).T
                b.attn.qkv.weight.data = torch.from_numpy(np.concatenate([qw,kw,vw])).float()
                qb = w[f"{ak}/query/bias"].reshape(-1)
                kb = w[f"{ak}/key/bias"].reshape(-1)
                vb = w[f"{ak}/value/bias"].reshape(-1)
                b.attn.qkv.bias.data = torch.from_numpy(np.concatenate([qb,kb,vb])).float(); n += 2
            if f"{ak}/out/kernel" in w:
                b.attn.proj.weight.data = torch.from_numpy(
                    w[f"{ak}/out/kernel"].reshape(-1,768).T).float()
                b.attn.proj.bias.data = torch.from_numpy(w[f"{ak}/out/bias"]).float(); n += 2
            if f"{p}/LayerNorm_2/scale" in w:
                b.norm2.weight.data = torch.from_numpy(w[f"{p}/LayerNorm_2/scale"]).float()
                b.norm2.bias.data = torch.from_numpy(w[f"{p}/LayerNorm_2/bias"]).float(); n += 2
            mk = f"{p}/MlpBlock_3"
            if f"{mk}/Dense_0/kernel" in w:
                b.ffn.fc1.weight.data = torch.from_numpy(w[f"{mk}/Dense_0/kernel"].T).float()
                b.ffn.fc1.bias.data = torch.from_numpy(w[f"{mk}/Dense_0/bias"]).float()
                b.ffn.fc2.weight.data = torch.from_numpy(w[f"{mk}/Dense_1/kernel"].T).float()
                b.ffn.fc2.bias.data = torch.from_numpy(w[f"{mk}/Dense_1/bias"]).float(); n += 4
        if "Transformer/encoder_norm/scale" in w:
            model.encoder.norm.weight.data = torch.from_numpy(
                w["Transformer/encoder_norm/scale"]).float()
            model.encoder.norm.bias.data = torch.from_numpy(
                w["Transformer/encoder_norm/bias"]).float(); n += 2

    print(f"  Loaded {n} weight tensors")
    print(f"  Note: 3D patch embed, PEFT params, and heads are randomly initialized")