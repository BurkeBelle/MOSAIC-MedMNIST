"""
UniMiSS Model Wrapper for Classification

Reproduce UniMiSS architecture from checkpoint:
- 5 stages: patch_embed0(32) -> block1(48) -> block2(128) -> block3(256) -> block4(512)
- Switchable Patch Embedding (SPE): Conv2d for 2D, Conv3d for 3D
- Each stage has independent sr2D/sr3D for spatial reduction
- depths: [2, 3, 4, 3] for block1-4

Reference: https://github.com/YtongXie/UniMiSS-code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    """MLP module"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self attention with switchable 2D/3D spatial reduction"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Spatial reduction for 2D and 3D (separate convolutions)
        if sr_ratio > 1:
            self.sr2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, D=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # Spatial reduction
            x_ = x[:, 1:].permute(0, 2, 1)  # B, C, N-1
            if D is None:  # 2D
                x_ = x_.reshape(B, C, H, W)
                x_ = self.sr2D(x_)
                x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            else:  # 3D
                x_ = x_.reshape(B, C, D, H, W)
                x_ = self.sr3D(x_)
                x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat([x[:, :1], x_], dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, D=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed2D(nn.Module):
    """2D Overlapping Patch Embedding (stage 0)"""
    def __init__(self, in_chans=3, embed_dim=32):
        super().__init__()
        # patch_embed2D0: kernel=7, stride=4, padding=3, no bias
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, H, W


class OverlapPatchEmbed3D(nn.Module):
    """3D Overlapping Patch Embedding (stage 0)"""
    def __init__(self, in_chans=1, embed_dim=32):
        super().__init__()
        # patch_embed3D0: kernel=7, stride=4, padding=3, no bias
        self.conv = nn.Conv3d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv(x)  # B, C, D, H, W
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, D, H, W


class PatchMerging2D(nn.Module):
    """2D Patch Merging between stages"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential()
        self.proj.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.proj.norm = nn.LayerNorm(out_dim)

    def forward(self, x, H, W, has_cls_token=False):
        B, N, C = x.shape
        # Remove cls token if present, reshape, apply conv
        if has_cls_token:
            x = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        else:
            x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj.conv(x)
        B, C_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj.norm(x)
        return x, H_new, W_new


class PatchMerging3D(nn.Module):
    """3D Patch Merging between stages"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential()
        self.proj.conv = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.proj.norm = nn.LayerNorm(out_dim)

    def forward(self, x, D, H, W, has_cls_token=False):
        B, N, C = x.shape
        if has_cls_token:
            x = x[:, 1:].transpose(1, 2).reshape(B, C, D, H, W)
        else:
            x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.proj.conv(x)
        B, C_new, D_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj.norm(x)
        return x, D_new, H_new, W_new


class UniMiSSTransformer(nn.Module):
    """
    UniMiSS Transformer Encoder
    
    Architecture based on checkpoint structure:
    - Stage 0: patch_embed (32 dim, stride 4)
    - Stage 1: block1 x 2, dim=48, sr=6
    - Stage 2: block2 x 3, dim=128, sr=4  
    - Stage 3: block3 x 4, dim=256, sr=2
    - Stage 4: block4 x 3, dim=512, sr=1 (no spatial reduction)
    """
    def __init__(self, 
                 in_chans_2d=3,
                 in_chans_3d=1,
                 embed_dims=[32, 48, 128, 256, 512],  # stage 0-4
                 num_heads=[1, 1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4, 4],
                 depths=[0, 2, 3, 4, 3],  # stage 0 has no blocks
                 sr_ratios=[0, 6, 4, 2, 1],  # stage 0 N/A, stage 4 = 1 (no sr)
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.depths = depths
        
        # Stage 0: Initial patch embedding
        self.patch_embed2D0 = OverlapPatchEmbed2D(in_chans=in_chans_2d, embed_dim=embed_dims[0])
        self.patch_embed3D0 = OverlapPatchEmbed3D(in_chans=in_chans_3d, embed_dim=embed_dims[0])
        
        # Patch merging between stages (2D and 3D)
        self.patch_embed2D1 = PatchMerging2D(embed_dims[0], embed_dims[1])
        self.patch_embed2D2 = PatchMerging2D(embed_dims[1], embed_dims[2])
        self.patch_embed2D3 = PatchMerging2D(embed_dims[2], embed_dims[3])
        self.patch_embed2D4 = PatchMerging2D(embed_dims[3], embed_dims[4])
        
        self.patch_embed3D1 = PatchMerging3D(embed_dims[0], embed_dims[1])
        self.patch_embed3D2 = PatchMerging3D(embed_dims[1], embed_dims[2])
        self.patch_embed3D3 = PatchMerging3D(embed_dims[2], embed_dims[3])
        self.patch_embed3D4 = PatchMerging3D(embed_dims[3], embed_dims[4])
        
        # CLS tokens: stage 1 is Parameter, stages 2-4 are Linear projections
        self.cls_tokens1 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_tokens2 = nn.Linear(embed_dims[1], embed_dims[2])
        self.cls_tokens3 = nn.Linear(embed_dims[2], embed_dims[3])
        self.cls_tokens4 = nn.Linear(embed_dims[3], embed_dims[4])
        
        # Position embeddings (2D and 3D) for each stage
        # Based on checkpoint: pos_embed2D1 [1, 3137, 48] = 56*56 + 1
        self.pos_embed2D1 = nn.Parameter(torch.zeros(1, 3137, embed_dims[1]))
        self.pos_embed2D2 = nn.Parameter(torch.zeros(1, 785, embed_dims[2]))   # 28*28 + 1
        self.pos_embed2D3 = nn.Parameter(torch.zeros(1, 197, embed_dims[3]))   # 14*14 + 1
        self.pos_embed2D4 = nn.Parameter(torch.zeros(1, 50, embed_dims[4]))    # 7*7 + 1
        
        self.pos_embed3D1 = nn.Parameter(torch.zeros(1, 4609, embed_dims[1]))  # 18*16*16 + 1
        self.pos_embed3D2 = nn.Parameter(torch.zeros(1, 577, embed_dims[2]))   # 9*8*8 + 1
        self.pos_embed3D3 = nn.Parameter(torch.zeros(1, 73, embed_dims[3]))    # 4*4*4 + 1
        self.pos_embed3D4 = nn.Parameter(torch.zeros(1, 10, embed_dims[4]))    # 2*2*2 + 1
        
        # Drop path
        total_depth = sum(depths[1:])  # exclude stage 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # Transformer blocks for stages 1-4
        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        cur += depths[3]
        
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[4], num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[4])
            for i in range(depths[4])])
        
        # Initialize
        trunc_normal_(self.cls_tokens1, std=.02)
        trunc_normal_(self.pos_embed2D1, std=.02)
        trunc_normal_(self.pos_embed2D2, std=.02)
        trunc_normal_(self.pos_embed2D3, std=.02)
        trunc_normal_(self.pos_embed2D4, std=.02)
        trunc_normal_(self.pos_embed3D1, std=.02)
        trunc_normal_(self.pos_embed3D2, std=.02)
        trunc_normal_(self.pos_embed3D3, std=.02)
        trunc_normal_(self.pos_embed3D4, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _interpolate_pos_embed(self, pos_embed, N, is_3d=False):
        """Interpolate position embedding if sequence length doesn't match"""
        if pos_embed.shape[1] == N:
            return pos_embed
        
        # Separate cls token and patch embeddings
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        num_patches = patch_pos.shape[1]
        dim = pos_embed.shape[-1]
        
        if not is_3d:
            # 2D: assume square
            old_size = int(num_patches ** 0.5)
            new_num_patches = N - 1
            new_size = int(new_num_patches ** 0.5)
            
            patch_pos = patch_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        else:
            # 3D: don't interpolate, just truncate or pad
            new_num_patches = N - 1
            if new_num_patches <= num_patches:
                # Truncate
                patch_pos = patch_pos[:, :new_num_patches]
            else:
                # Pad with zeros
                padding = torch.zeros(1, new_num_patches - num_patches, dim, device=patch_pos.device, dtype=patch_pos.dtype)
                patch_pos = torch.cat([patch_pos, padding], dim=1)
        
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_2d(self, x):
        """Forward for 2D input"""
        B = x.shape[0]
        
        # Stage 0: Initial patch embedding (no cls token yet)
        x, H, W = self.patch_embed2D0(x)  # B, H*W, 32
        
        # Stage 1: Downsample (no cls token) + add cls token + transformer blocks
        x, H, W = self.patch_embed2D1(x, H, W, has_cls_token=False)  # B, H*W, 48
        cls_token = self.cls_tokens1.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding (interpolate if needed)
        pos_embed = self._interpolate_pos_embed(self.pos_embed2D1, x.shape[1], is_3d=False)
        x = x + pos_embed
        
        for blk in self.block1:
            x = blk(x, H, W)
        
        # Stage 2: has cls token now
        cls_token = x[:, :1]
        x, H, W = self.patch_embed2D2(x, H, W, has_cls_token=True)
        cls_token = self.cls_tokens2(cls_token)
        x = torch.cat([cls_token, x], dim=1)
        pos_embed = self._interpolate_pos_embed(self.pos_embed2D2, x.shape[1], is_3d=False)
        x = x + pos_embed
        
        for blk in self.block2:
            x = blk(x, H, W)
        
        # Stage 3
        cls_token = x[:, :1]
        x, H, W = self.patch_embed2D3(x, H, W, has_cls_token=True)
        cls_token = self.cls_tokens3(cls_token)
        x = torch.cat([cls_token, x], dim=1)
        pos_embed = self._interpolate_pos_embed(self.pos_embed2D3, x.shape[1], is_3d=False)
        x = x + pos_embed
        
        for blk in self.block3:
            x = blk(x, H, W)
        
        # Stage 4
        cls_token = x[:, :1]
        x, H, W = self.patch_embed2D4(x, H, W, has_cls_token=True)
        cls_token = self.cls_tokens4(cls_token)
        x = torch.cat([cls_token, x], dim=1)
        pos_embed = self._interpolate_pos_embed(self.pos_embed2D4, x.shape[1], is_3d=False)
        x = x + pos_embed
        
        for blk in self.block4:
            x = blk(x, H, W)
        
        # Return CLS token
        return x[:, 0]

    def forward_3d(self, x):
        """Forward for 3D input"""
        B = x.shape[0]
        
        # Stage 0 (no cls token)
        x, D, H, W = self.patch_embed3D0(x)
        
        # Stage 1: Downsample (no cls token) + add cls token
        x, D, H, W = self.patch_embed3D1(x, D, H, W, has_cls_token=False)
        cls_token = self.cls_tokens1.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        pos_embed = self._interpolate_pos_embed(self.pos_embed3D1, x.shape[1], is_3d=True)
        x = x + pos_embed
        
        for blk in self.block1:
            x = blk(x, H, W, D)
        
        # Stage 2: has cls token now
        cls_token = x[:, :1]
        x, D, H, W = self.patch_embed3D2(x, D, H, W, has_cls_token=True)
        cls_token = self.cls_tokens2(cls_token)
        x = torch.cat([cls_token, x], dim=1)
        pos_embed = self._interpolate_pos_embed(self.pos_embed3D2, x.shape[1], is_3d=True)
        x = x + pos_embed
        
        for blk in self.block2:
            x = blk(x, H, W, D)
        
        # Stage 3
        cls_token = x[:, :1]
        x, D, H, W = self.patch_embed3D3(x, D, H, W, has_cls_token=True)
        cls_token = self.cls_tokens3(cls_token)
        x = torch.cat([cls_token, x], dim=1)
        pos_embed = self._interpolate_pos_embed(self.pos_embed3D3, x.shape[1], is_3d=True)
        x = x + pos_embed
        
        for blk in self.block3:
            x = blk(x, H, W, D)
        
        # Stage 4
        cls_token = x[:, :1]
        x, D, H, W = self.patch_embed3D4(x, D, H, W, has_cls_token=True)
        cls_token = self.cls_tokens4(cls_token)
        x = torch.cat([cls_token, x], dim=1)
        pos_embed = self._interpolate_pos_embed(self.pos_embed3D4, x.shape[1], is_3d=True)
        x = x + pos_embed
        
        for blk in self.block4:
            x = blk(x, H, W, D)
        
        return x[:, 0]

    def forward(self, x):
        """Auto switch between 2D and 3D"""
        if x.dim() == 4:  # B, C, H, W
            return self.forward_2d(x)
        elif x.dim() == 5:  # B, C, D, H, W
            return self.forward_3d(x)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")


class UniMiSSClassifier(nn.Module):
    """
    UniMiSS Classifier Wrapper
    
    Load pretrained encoder weights + classification head
    """
    def __init__(self, 
                 num_classes=2,
                 pretrained_path=None,
                 embed_dim=512,  # last stage dimension
                 dropout=0.1,
                 in_chans_2d=3,
                 in_chans_3d=1):
        super().__init__()
        
        self.transformer = UniMiSSTransformer(in_chans_2d=in_chans_2d, in_chans_3d=in_chans_3d)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, path):
        """Load pretrained weights from UniMiSS checkpoint"""
        print(f"Loading UniMiSS weights from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        # Use teacher weights (typically more stable)
        if 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        elif 'student' in checkpoint:
            state_dict = checkpoint['student']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        # Extract backbone.transformer weights
        transformer_state = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.transformer.'):
                new_key = k.replace('backbone.transformer.', '')
                transformer_state[new_key] = v
        
        # Load weights
        missing, unexpected = self.transformer.load_state_dict(transformer_state, strict=False)
        
        print(f"Loaded {len(transformer_state)} keys from checkpoint")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        
        if missing:
            print(f"Sample missing: {missing[:10]}")
        if unexpected:
            print(f"Sample unexpected: {unexpected[:10]}")
    
    def forward(self, x):
        features = self.transformer(x)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits
    
    def get_features(self, x):
        """Extract features without classification head"""
        return self.transformer(x)


def create_unimiss_classifier(num_classes, pretrained_path=None, in_chans_2d=3, in_chans_3d=1):
    """Factory function to create UniMiSS classifier"""
    return UniMiSSClassifier(
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        embed_dim=512,
        dropout=0.1,
        in_chans_2d=in_chans_2d,
        in_chans_3d=in_chans_3d
    )


# Test code
if __name__ == "__main__":
    print("Testing UniMiSS model...")
    
    # Test 2D
    model = UniMiSSClassifier(num_classes=2)
    x_2d = torch.randn(2, 3, 224, 224)
    out_2d = model(x_2d)
    print(f"2D input: {x_2d.shape} -> output: {out_2d.shape}")
    
    # Test 3D
    x_3d = torch.randn(2, 1, 64, 64, 64)
    out_3d = model(x_3d)
    print(f"3D input: {x_3d.shape} -> output: {out_3d.shape}")
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")