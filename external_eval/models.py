# external_eval/models.py
"""
Model loading module

Supports:
1. ImageNet ViT-Base
2. RadImageNet ResNet50
3. RadImageNet DenseNet121
4. RadImageNet InceptionV3
5. Ours (ViT-Base + MoE Adapter)

Each model:
1. Load pretrained weights
2. Freeze backbone
3. Add new classification head
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple

from configs import MODELS, WEIGHTS_ROOT, DATASETS

# Add project root to path to import model modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Base class: model with classification head
# ============================================================
class BaseModelWithHead(nn.Module):
    """
    Base class: backbone + classification head
    """
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize classification head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    
    def freeze_backbone(self):
        """Freeze backbone, train classification head only"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Ensure classification head is trainable
        for param in self.head.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 1. ImageNet ViT-Base
# ============================================================
class ImageNetViT(BaseModelWithHead):
    """
    ImageNet pretrained ViT-Base
    Loaded via timm
    """
    def __init__(self, num_classes: int, weights_path: str):
        import timm
        
        # Load ViT-Base model (without pretrained weights)
        backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        
        # Load ImageNet pretrained weights (npz format)
        self._load_npz_weights(backbone, weights_path)
        
        super().__init__(backbone, embed_dim=768, num_classes=num_classes)
        print(f"[ImageNetViT] Loaded weights from {weights_path}")
        print(f"[ImageNetViT] num_classes={num_classes}, embed_dim=768")
    
    def _load_npz_weights(self, model, npz_path: str):
        """Load ViT weights from npz file"""
        weights = np.load(npz_path)
        
        with torch.no_grad():
            # Patch Embedding
            if 'embedding/kernel' in weights:
                kernel = weights['embedding/kernel']
                kernel = np.transpose(kernel, (3, 2, 0, 1))
                model.patch_embed.proj.weight.data = torch.from_numpy(kernel).float()
            if 'embedding/bias' in weights:
                model.patch_embed.proj.bias.data = torch.from_numpy(weights['embedding/bias']).float()
            
            # CLS token
            if 'cls' in weights:
                model.cls_token.data = torch.from_numpy(weights['cls']).float()
            
            # Position Embedding
            if 'Transformer/posembed_input/pos_embedding' in weights:
                pos_embed = weights['Transformer/posembed_input/pos_embedding']
                if pos_embed.shape[1] == model.pos_embed.shape[1]:
                    model.pos_embed.data = torch.from_numpy(pos_embed).float()
            
            # Transformer Blocks
            for block_idx in range(12):
                prefix = f'Transformer/encoderblock_{block_idx}'
                block = model.blocks[block_idx]
                
                # LayerNorm 1
                if f'{prefix}/LayerNorm_0/scale' in weights:
                    block.norm1.weight.data = torch.from_numpy(weights[f'{prefix}/LayerNorm_0/scale']).float()
                    block.norm1.bias.data = torch.from_numpy(weights[f'{prefix}/LayerNorm_0/bias']).float()
                
                # Attention QKV
                if f'{prefix}/MultiHeadDotProductAttention_1/query/kernel' in weights:
                    q_w = weights[f'{prefix}/MultiHeadDotProductAttention_1/query/kernel']
                    k_w = weights[f'{prefix}/MultiHeadDotProductAttention_1/key/kernel']
                    v_w = weights[f'{prefix}/MultiHeadDotProductAttention_1/value/kernel']
                    
                    q_w = q_w.reshape(q_w.shape[0], -1).T
                    k_w = k_w.reshape(k_w.shape[0], -1).T
                    v_w = v_w.reshape(v_w.shape[0], -1).T
                    qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
                    block.attn.qkv.weight.data = torch.from_numpy(qkv_w).float()
                    
                    q_b = weights[f'{prefix}/MultiHeadDotProductAttention_1/query/bias'].reshape(-1)
                    k_b = weights[f'{prefix}/MultiHeadDotProductAttention_1/key/bias'].reshape(-1)
                    v_b = weights[f'{prefix}/MultiHeadDotProductAttention_1/value/bias'].reshape(-1)
                    qkv_b = np.concatenate([q_b, k_b, v_b], axis=0)
                    block.attn.qkv.bias.data = torch.from_numpy(qkv_b).float()
                
                # Attention output projection
                if f'{prefix}/MultiHeadDotProductAttention_1/out/kernel' in weights:
                    out_w = weights[f'{prefix}/MultiHeadDotProductAttention_1/out/kernel']
                    out_w = out_w.reshape(-1, out_w.shape[-1]).T
                    block.attn.proj.weight.data = torch.from_numpy(out_w).float()
                    block.attn.proj.bias.data = torch.from_numpy(
                        weights[f'{prefix}/MultiHeadDotProductAttention_1/out/bias']
                    ).float()
                
                # LayerNorm 2
                if f'{prefix}/LayerNorm_2/scale' in weights:
                    block.norm2.weight.data = torch.from_numpy(weights[f'{prefix}/LayerNorm_2/scale']).float()
                    block.norm2.bias.data = torch.from_numpy(weights[f'{prefix}/LayerNorm_2/bias']).float()
                
                # MLP
                if f'{prefix}/MlpBlock_3/Dense_0/kernel' in weights:
                    block.mlp.fc1.weight.data = torch.from_numpy(
                        weights[f'{prefix}/MlpBlock_3/Dense_0/kernel'].T
                    ).float()
                    block.mlp.fc1.bias.data = torch.from_numpy(
                        weights[f'{prefix}/MlpBlock_3/Dense_0/bias']
                    ).float()
                
                if f'{prefix}/MlpBlock_3/Dense_1/kernel' in weights:
                    block.mlp.fc2.weight.data = torch.from_numpy(
                        weights[f'{prefix}/MlpBlock_3/Dense_1/kernel'].T
                    ).float()
                    block.mlp.fc2.bias.data = torch.from_numpy(
                        weights[f'{prefix}/MlpBlock_3/Dense_1/bias']
                    ).float()
            
            # Final LayerNorm
            if 'Transformer/encoder_norm/scale' in weights:
                model.norm.weight.data = torch.from_numpy(weights['Transformer/encoder_norm/scale']).float()
                model.norm.bias.data = torch.from_numpy(weights['Transformer/encoder_norm/bias']).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [B, 768]
        logits = self.head(features)  # [B, num_classes]
        return logits


# ============================================================
# 2. RadImageNet ResNet50
# ============================================================
class RadImageNetResNet50(BaseModelWithHead):
    """
    RadImageNet pretrained ResNet50
    """
    def __init__(self, num_classes: int, weights_path: str):
        # Create ResNet50
        backbone = models.resnet50(pretrained=False)
        
        # Remove original head, get feature dim
        embed_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        
        # Load RadImageNet weights
        self._load_radimagenet_weights(backbone, weights_path)
        
        super().__init__(backbone, embed_dim=embed_dim, num_classes=num_classes)
        print(f"[RadImageNetResNet50] Loaded weights from {weights_path}")
        print(f"[RadImageNetResNet50] num_classes={num_classes}, embed_dim={embed_dim}")
    
    def _load_radimagenet_weights(self, model, weights_path: str):
        """Load RadImageNet weights (handle backbone. prefix)"""
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # RadImageNet ResNet50 key mapping:
        # backbone.0 -> conv1
        # backbone.1 -> bn1
        # backbone.4 -> layer1
        # backbone.5 -> layer2
        # backbone.6 -> layer3
        # backbone.7 -> layer4
        
        mapping = {
            'backbone.0.': 'conv1.',
            'backbone.1.': 'bn1.',
            'backbone.4.': 'layer1.',
            'backbone.5.': 'layer2.',
            'backbone.6.': 'layer3.',
            'backbone.7.': 'layer4.',
        }
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for old_prefix, new_prefix in mapping.items():
                if k.startswith(old_prefix):
                    new_key = k.replace(old_prefix, new_prefix)
                    break
            new_state_dict[new_key] = v
        
        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"  Loaded {len(new_state_dict) - len(missing)}/{len(new_state_dict)} keys")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [B, 2048]
        logits = self.head(features)  # [B, num_classes]
        return logits


# ============================================================
# 3. RadImageNet DenseNet121
# ============================================================
class RadImageNetDenseNet121(BaseModelWithHead):
    """
    RadImageNet pretrained DenseNet121
    """
    def __init__(self, num_classes: int, weights_path: str):
        # Create DenseNet121
        backbone = models.densenet121(pretrained=False)
        
        # Get feature dim and remove head
        embed_dim = backbone.classifier.in_features  # 1024
        backbone.classifier = nn.Identity()
        
        # Load RadImageNet weights
        self._load_radimagenet_weights(backbone, weights_path)
        
        super().__init__(backbone, embed_dim=embed_dim, num_classes=num_classes)
        print(f"[RadImageNetDenseNet121] Loaded weights from {weights_path}")
        print(f"[RadImageNetDenseNet121] num_classes={num_classes}, embed_dim={embed_dim}")
    
    def _load_radimagenet_weights(self, model, weights_path: str):
        """Load RadImageNet weights"""
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # RadImageNet DenseNet121 key mapping:
        # backbone.0.xxx -> features.xxx
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.0.'):
                new_key = k.replace('backbone.0.', 'features.')
                new_state_dict[new_key] = v
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"  Loaded {len(new_state_dict) - len(missing)}/{len(new_state_dict)} keys")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [B, 1024]
        logits = self.head(features)  # [B, num_classes]
        return logits


# ============================================================
# 4. RadImageNet InceptionV3
# ============================================================
class RadImageNetInceptionV3(BaseModelWithHead):
    """
    RadImageNet pretrained InceptionV3
    """
    def __init__(self, num_classes: int, weights_path: str):
        # Create InceptionV3 (default input 299x299, we use 224x224)
        backbone = models.inception_v3(pretrained=False, aux_logits=False)
        
        # Get feature dim and remove head
        embed_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        
        # Load RadImageNet weights
        self._load_radimagenet_weights(backbone, weights_path)
        
        super().__init__(backbone, embed_dim=embed_dim, num_classes=num_classes)
        print(f"[RadImageNetInceptionV3] Loaded weights from {weights_path}")
        print(f"[RadImageNetInceptionV3] num_classes={num_classes}, embed_dim={embed_dim}")
    
    def _load_radimagenet_weights(self, model, weights_path: str):
        """Load RadImageNet weights"""
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # RadImageNet InceptionV3 key mapping:
        # backbone.0 -> Conv2d_1a_3x3
        # backbone.1 -> Conv2d_2a_3x3
        # backbone.2 -> Conv2d_2b_3x3
        # backbone.4 -> Conv2d_3b_1x1
        # backbone.5 -> Conv2d_4a_3x3
        # backbone.7 -> Mixed_5b
        # backbone.8 -> Mixed_5c
        # backbone.9 -> Mixed_5d
        # backbone.10 -> Mixed_6a
        # backbone.11 -> Mixed_6b
        # backbone.12 -> Mixed_6c
        # backbone.13 -> Mixed_6d
        # backbone.14 -> Mixed_6e
        # backbone.15 -> Mixed_7a
        # backbone.16 -> Mixed_7b
        # backbone.17 -> Mixed_7c
        
        mapping = {
            'backbone.0.': 'Conv2d_1a_3x3.',
            'backbone.1.': 'Conv2d_2a_3x3.',
            'backbone.2.': 'Conv2d_2b_3x3.',
            'backbone.4.': 'Conv2d_3b_1x1.',
            'backbone.5.': 'Conv2d_4a_3x3.',
            'backbone.7.': 'Mixed_5b.',
            'backbone.8.': 'Mixed_5c.',
            'backbone.9.': 'Mixed_5d.',
            'backbone.10.': 'Mixed_6a.',
            'backbone.11.': 'Mixed_6b.',
            'backbone.12.': 'Mixed_6c.',
            'backbone.13.': 'Mixed_6d.',
            'backbone.14.': 'Mixed_6e.',
            'backbone.15.': 'Mixed_7a.',
            'backbone.16.': 'Mixed_7b.',
            'backbone.17.': 'Mixed_7c.',
        }
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for old_prefix, new_prefix in mapping.items():
                if k.startswith(old_prefix):
                    new_key = k.replace(old_prefix, new_prefix)
                    break
            new_state_dict[new_key] = v
        
        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"  Loaded {len(new_state_dict) - len(missing)}/{len(new_state_dict)} keys")
        if len(missing) > 0:
            print(f"  Missing: {len(missing)} keys")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # InceptionV3 expects 299x299 input; we use 224x224
        # Using 224x224 directly (minor accuracy difference possible)
        features = self.backbone(x)  # [B, 2048]
        logits = self.head(features)  # [B, num_classes]
        return logits


# ============================================================
# 5. Ours (ViT-Base + MoE Adapter)
# ============================================================
class OursViTMoE(nn.Module):
    """
    Our model: ViT-Base + MoE Adapter
    """
    def __init__(self, num_classes: int, weights_path: str):
        super().__init__()  # call parent init first
        
        # Import our model
        from model.unified_model import UnifiedModel
        
        # MedMNIST 18-dataset class counts (must match checkpoint)
        medmnist_classes = [9, 7, 4, 2, 14, 2, 8, 8, 5, 11, 11, 11, 11, 2, 2, 2, 3, 2]
        
        # Create model (same config as checkpoint)
        backbone = UnifiedModel(
            num_classes_list=medmnist_classes,  # 18 tasks
            embed_dim=768,
            depth=12,
            num_heads=12,
            use_adapter=True,
            adapter_mode='v2_moe',
            adapter_bottleneck_a=64,
            adapter_bottleneck_b=96,
            adapter_bottleneck_c=192,  # Note: 192 (not 128)
            adapter_scalar=0.1
        )
        
        # Load pretrained weights
        self._load_checkpoint(backbone, weights_path)
        
        # Save backbone components
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed_2d = backbone.pos_embed_2d
        self.encoder = backbone.encoder
        
        # New classification head
        self.embed_dim = 768
        self.head = nn.Linear(768, num_classes)
        
        # Initialize classification head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        
        print(f"[OursViTMoE] Loaded weights from {weights_path}")
        print(f"[OursViTMoE] num_classes={num_classes}, embed_dim=768")
    
    def _load_checkpoint(self, model, weights_path: str):
        """Load checkpoint (extract from student_state_dict)"""
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if 'student_state_dict' in checkpoint:
            state_dict = checkpoint['student_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights (ignore classification heads)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint, missing: {len(missing)}, unexpected: {len(unexpected)}")
    
    def freeze_backbone(self):
        """Freeze backbone, train classification head only"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed_2d.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Classification head is trainable
        for param in self.head.parameters():
            param.requires_grad = True
    
    def freeze_backbone_keep_adapter(self):
        """Freeze ViT backbone, keep adapter and classification head trainable
        
        Standard adapter tuning approach:
        - ViT attention, FFN, etc. are frozen
        - Adapter parameters are trainable
        - Classification head is trainable
        """
        # Freeze all parameters first
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed_2d.requires_grad = False
        
        # Iterate encoder, freeze non-adapter parameters
        for name, param in self.encoder.named_parameters():
            if 'adapter' in name.lower():
                param.requires_grad = True  # Adapter trainable
            else:
                param.requires_grad = False  # others frozen
        
        # Classification head is trainable
        for param in self.head.parameters():
            param.requires_grad = True
        
        # Print trainable parameter info
        adapter_params = sum(p.numel() for n, p in self.encoder.named_parameters() 
                            if 'adapter' in n.lower() and p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f"  [Adapter Tuning] Adapter params: {adapter_params:,}, Head params: {head_params:,}")
        print(f"  [Adapter Tuning] Total trainable: {adapter_params + head_params:,}")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone"""
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        self.cls_token.requires_grad = True
        self.pos_embed_2d.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        count = 0
        for param in self.patch_embed.parameters():
            if param.requires_grad:
                count += param.numel()
        if self.cls_token.requires_grad:
            count += self.cls_token.numel()
        if self.pos_embed_2d.requires_grad:
            count += self.pos_embed_2d.numel()
        for param in self.encoder.parameters():
            if param.requires_grad:
                count += param.numel()
        for param in self.head.parameters():
            if param.requires_grad:
                count += param.numel()
        return count
    
    def get_total_params(self):
        """Get total number of parameters"""
        count = 0
        for param in self.patch_embed.parameters():
            count += param.numel()
        count += self.cls_token.numel()
        count += self.pos_embed_2d.numel()
        for param in self.encoder.parameters():
            count += param.numel()
        for param in self.head.parameters():
            count += param.numel()
        return count
    
    def set_expert(self, expert_id: str):
        """Set which expert to use
        
        Args:
            expert_id: 'A' (Bio-Medical RGB), 'B' (Radiology Grayscale), 'C' (3D Volumetric)
        """
        assert expert_id in ['A', 'B', 'C'], f"Invalid expert_id: {expert_id}"
        self.expert_id = expert_id
        print(f"  [OursViTMoE] Using Expert {expert_id}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch Embedding (2D only)
        tokens, is_3d = self.patch_embed(x)
        B = tokens.shape[0]
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        
        # Add positional encoding
        tokens = tokens + self.pos_embed_2d
        
        # Transformer Encoder
        # Use configured expert_id, default 'A'
        expert_id = getattr(self, 'expert_id', 'A')
        tokens = self.encoder(tokens, is_3d=False, expert_id=expert_id)
        
        # Extract CLS token
        features = tokens[:, 0]  # [B, 768]
        
        # Classification
        logits = self.head(features)  # [B, num_classes]
        
        return logits


# ============================================================
# Model factory
# ============================================================
def create_model(model_name: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Create model
    
    Args:
        model_name: model name (imagenet_vit, radimagenet_resnet50, etc.)
        num_classes: number of classes
        freeze_backbone: whether to freeze backbone
    
    Returns:
        model: model with classification head
    """
    config = MODELS[model_name]
    weights_path = config['weights']
    
    if model_name == 'imagenet_vit':
        model = ImageNetViT(num_classes, weights_path)
    elif model_name == 'radimagenet_resnet50':
        model = RadImageNetResNet50(num_classes, weights_path)
    elif model_name == 'radimagenet_densenet121':
        model = RadImageNetDenseNet121(num_classes, weights_path)
    elif model_name == 'radimagenet_inceptionv3':
        model = RadImageNetInceptionV3(num_classes, weights_path)
    elif model_name == 'ours':
        model = OursViTMoE(num_classes, weights_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if freeze_backbone:
        model.freeze_backbone()
        print(f"  Backbone frozen. Trainable params: {model.get_trainable_params():,}")
    else:
        print(f"  Backbone unfrozen. Trainable params: {model.get_trainable_params():,}")
    
    print(f"  Total params: {model.get_total_params():,}")
    
    return model


# ============================================================
# Get available model list（excluding MedCoSS）
# ============================================================
def get_available_models():
    """Get available model list"""
    return [
        'imagenet_vit',
        'radimagenet_resnet50',
        'radimagenet_densenet121',
        'radimagenet_inceptionv3',
        'ours'
    ]


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Models test")
    print("=" * 60)
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    
    for model_name in get_available_models():
        print(f"\n[Testing {model_name}]")
        try:
            model = create_model(model_name, num_classes=2, freeze_backbone=True)
            model.eval()
            
            with torch.no_grad():
                output = model(x)
            
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  ✓ Success!")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)