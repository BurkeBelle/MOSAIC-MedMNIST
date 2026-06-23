# external_eval/models_3d.py
"""
3D External Validation model definitions

Contains:
1. Med3D 3D-ResNet (18/34/50) - load pretrained weights from MedicalNet
2. ImageNet ViT 3D - UnifiedModel(no adapter)load ImageNet weights, 3D tokenizer randomly initialized
3. Ours Expert C - UnifiedModel + MoE Adapter (Expert C for 3D volumetric)

Each model:
  Input: (B, 1, 64, 64, 64) grayscale CT volume
  Output: backbone features -> linear head -> (B, num_classes)

Key design:
- ImageNet ViT uses UnifiedModel because it already has 3D patch embedding + positional encoding
  and can handle 5D input directly, with adapter disabled for pure backbone baseline
- Ours Expert C enables V2 MoE Adapter, using Expert C for 3D volumetric data
- Both extract CLS token via forward_features(), then linear classification head
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs_3d import MODELS_3D, DEVICE

# Add project root to path to import unified_model
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 1. Med3D 3D-ResNet series
# ============================================================
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class BasicBlock3D(nn.Module):
    """3D ResNet BasicBlock (ResNet-18/34)"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck3D(nn.Module):
    """3D ResNet Bottleneck (ResNet-50/101/152)"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    """
    Med3D 3D-ResNet

    Architecture reproduced from Tencent/MedicalNet:
    - conv1: (1, 64, 7×7×7, stride=2)
    - bn1 + relu + maxpool
    - layer1/2/3/4
    - avgpool -> fc
    """

    def __init__(self, block, layers, shortcut_type='B',
                 num_classes=2, in_channels=1):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7,
                               stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.embed_dim = 512 * block.expansion
        self.fc = nn.Linear(self.embed_dim, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = nn.Sequential(
                    conv1x1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:  # shortcut_type == 'B'
                downsample = nn.Sequential(
                    conv1x1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        """Extract features (without fc)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        out = self.fc(features)
        return out


def _resnet3d(depth, shortcut_type='B', **kwargs):
    """Create 3D-ResNet of specified depth"""
    configs = {
        18: (BasicBlock3D, [2, 2, 2, 2]),
        34: (BasicBlock3D, [3, 4, 6, 3]),
        50: (Bottleneck3D, [3, 4, 6, 3]),
    }
    if depth not in configs:
        raise ValueError(f"Unsupported depth: {depth}. Choose from {list(configs.keys())}")
    block, layers = configs[depth]
    return ResNet3D(block, layers, shortcut_type=shortcut_type, **kwargs)


def load_med3d_weights(model, weights_path):
    """
    Load Med3D pretrained weights

    Weight format: {'state_dict': {module.xxx: tensor, ...}}
    Need to remove 'module.' prefix
    """
    checkpoint = torch.load(weights_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix (DataParallel format)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    # Load backbone weights only, skip task heads like fc/conv_seg
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}

    n_loaded = len(pretrained_dict)
    n_total = len(model_dict)
    print(f"  Med3D: loaded {n_loaded}/{n_total} parameters from {os.path.basename(weights_path)}")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


# ============================================================
# 2. ImageNet ViT 3D (UnifiedModel backbone, no adapter)
# ============================================================
class ImageNetViT3D(nn.Module):
    """
    ImageNet ViT-Base for 3D input

    Architecture: UnifiedModel (no adapter, pure ViT backbone)
    Weights: ImageNet npz -> backbone (2D patch embed, transformer, cls token)
           3D tokenizer (patch_embed_3d, pos_embed_3d) randomly initialized

    This is the general pretrained baseline for 3D, counterpart of 2D ImageNet ViT.
    Uses UnifiedModel because it already has 3D patch embedding + positional encoding,
    and can directly process (B,1,64,64,64) input.
    """

    def __init__(self, num_classes=2, weights_path=None):
        super().__init__()
        from model.unified_model import UnifiedModel, load_pretrained_vit_from_npz

        # Create UnifiedModel with adapter disabled (pure backbone)
        # num_classes_list is dummy, we don't use its heads
        self.backbone = UnifiedModel(
            num_classes_list=[num_classes],  # dummy, not used
            img_size_2d=224,
            img_size_3d=64,
            in_chans_2d=3,
            in_chans_3d=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            use_adapter=False,  # pure backbone, no adapter
        )
        self.embed_dim = 768

        # Use existing npz loader with full key mapping
        if weights_path and os.path.exists(weights_path):
            load_pretrained_vit_from_npz(self.backbone, weights_path)
            print(f"  Note: 3D tokenizer (patch_embed_3d, pos_embed_3d) randomly initialized")

        # External classification head (linear probing)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x):
        """Extract 3D features, return CLS token [B, 768]"""
        # forward_features auto-detects input dim (4D=2D, 5D=3D)
        features, is_3d = self.backbone.forward_features(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        out = self.fc(features)
        return out


# ============================================================
# 3. Ours: Expert C (UnifiedModel + MoE Adapter)
# ============================================================
# ============================================================
# 3. Ours: Expert C (UnifiedModel + MoE Adapter)
# ============================================================
class OursExpertC3D(nn.Module):
    """
    Our method: UnifiedModel + MoE Expert C

    Architecture: UnifiedModel + V2 MoE Adapter (3 experts)
    Weights: trained best_model.pth (full model params)
    Inference: Expert C (3D volumetric specialist)

    Counterpart of 2D "Ours" model.
    """

    def __init__(self, num_classes=2, weights_path=None,
                 adapter_bottleneck_a=64, adapter_bottleneck_b=96,
                 adapter_bottleneck_c=192, adapter_scalar=0.1):
        super().__init__()
        from model.unified_model import UnifiedModel

        # Create UnifiedModel with MoE Adapter (same as training)
        # num_classes_list is dummy, internal heads not used
        # MedMNIST 18-dataset class counts
        medmnist_classes = [
            9, 7, 4, 2, 14, 2, 8, 8, 5, 11, 11, 11,  # 12 2D
            11, 2, 2, 2, 3, 2,                          # 6 3D
        ]
        self.backbone = UnifiedModel(
            num_classes_list=medmnist_classes,
            img_size_2d=224,
            img_size_3d=64,
            in_chans_2d=3,
            in_chans_3d=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            use_adapter=True,
            adapter_mode='v2_moe',
            adapter_bottleneck_a=adapter_bottleneck_a,
            adapter_bottleneck_b=adapter_bottleneck_b,
            adapter_bottleneck_c=adapter_bottleneck_c,
            adapter_scalar=adapter_scalar,
        )
        self.embed_dim = 768

        # Load full checkpoint
        if weights_path and os.path.exists(weights_path):
            self._load_checkpoint(weights_path)

        # External classification head (linear probing)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def _load_checkpoint(self, ckpt_path):
        """Load trained checkpoint"""
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'student_state_dict' in checkpoint:
                state_dict = checkpoint['student_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove possible 'module.' prefix (DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        # Load into backbone (strict=False: head keys may not match)
        msg = self.backbone.load_state_dict(new_state_dict, strict=False)
        n_loaded = len(new_state_dict) - len(msg.unexpected_keys)
        print(f"  Ours Expert C: loaded {n_loaded} parameters from {os.path.basename(ckpt_path)}")
        if msg.missing_keys:
            n_miss = len(msg.missing_keys)
            print(f"  Missing: {n_miss} keys (expected if classification heads differ)")
        if msg.unexpected_keys:
            n_unexp = len(msg.unexpected_keys)
            print(f"  Unexpected: {n_unexp} keys (ignored)")

    def forward_features(self, x):
        """Extract 3D features using Expert C, return CLS token [B, 768]"""
        # expert_id='C' selects 3D volumetric expert
        features, is_3d = self.backbone.forward_features(x, expert_id='C')
        return features

    def forward(self, x):
        features = self.forward_features(x)
        out = self.fc(features)
        return out


# ============================================================
# Model factory
# ============================================================
def create_model_3d(model_name, num_classes=2, freeze_backbone=True, adapter_tuning=False):
    """
    Create 3D model

    Args:
        model_name: model name (key in configs_3d.MODELS_3D)
        num_classes: number of classes
        freeze_backbone: whether to freeze backbone
        adapter_tuning: unfreeze adapter+fc only (Ours model)

    Returns:
        model: created model (moved to DEVICE)
    """
    cfg = MODELS_3D[model_name]
    arch = cfg['arch']
    weights = cfg['weights']

    print(f"\n--- Creating: {cfg['display_name']} ---")
    print(f"  Architecture: {arch}")
    print(f"  Weights: {os.path.basename(weights)}")

    # --- Med3D 3D-ResNet ---
    if arch == 'resnet3d':
        depth = cfg['model_depth']
        shortcut = cfg['resnet_shortcut']
        model = _resnet3d(depth, shortcut_type=shortcut,
                          num_classes=num_classes, in_channels=1)
        if os.path.exists(weights):
            model = load_med3d_weights(model, weights)
        else:
            print(f"  WARNING: weights not found at {weights}")
        # Re-initialize classification head (Med3D was originally a segmentation model)
        model.fc = nn.Linear(model.embed_dim, num_classes)

    # --- ImageNet ViT 3D ---
    elif arch == 'unified_model':
        model = ImageNetViT3D(num_classes=num_classes, weights_path=weights)

    # --- Ours Expert C ---
    elif arch == 'unified_model_moe':
        model = OursExpertC3D(num_classes=num_classes, weights_path=weights)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Freeze strategy
    if adapter_tuning and arch == 'unified_model_moe':
        # Adapter tuning: freeze backbone, train adapter+fc
        _freeze_adapter_tuning(model)
    elif freeze_backbone:
        _freeze_backbone(model, arch)

    # Print parameter statistics
    _print_param_stats(model)

    model = model.to(DEVICE)
    return model


def _freeze_backbone(model, arch):
    """Freeze backbone, train classification head only"""
    if arch == 'resnet3d':
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    elif arch in ('unified_model', 'unified_model_moe'):
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False


def _freeze_adapter_tuning(model):
    """
    Adapter tuning: freeze backbone transformer, train adapter+fc

    Unfrozen params (containing these keywords):
    - 'adapter': MoE adapter down/up projection
    - 'fc': external classification head
    - 'patch_embed_3d': 3D tokenizer (optional, randomly initialized)
    - 'pos_embed_3d': 3D positional encoding

    Frozen params:
    - transformer blocks (attention, mlp, norm)
    - 2D patch embedding
    - cls_token, pos_embed (2D)
    """
    # Note: can't use 'fc' as substring since ffn.fc1/fc2 also contain 'fc'
    # External head params: 'fc.weight' and 'fc.bias' (without 'backbone.')
    adapter_keywords = ['adapter', 'patch_embed_3d', 'pos_embed_3d', 'norm']

    trainable_names = []
    for name, param in model.named_parameters():
        # External head: exact match (name starts with 'fc.', not 'ffn')
        is_fc_head = name.startswith('fc.') or name == 'fc.weight' or name == 'fc.bias'
        is_adapter_related = any(kw in name for kw in adapter_keywords)

        if is_fc_head or is_adapter_related:
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False

    print(f"  [Adapter tuning] Trainable params ({len(trainable_names)}):")
    for n in trainable_names:
        p = dict(model.named_parameters())[n]
        print(f"    {n}: {list(p.shape)}")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {total_trainable:,} / {total_all:,} ({100*total_trainable/total_all:.2f}%)")


def _print_param_stats(model):
    """Print parameter statistics"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  Total params: {total:,} ({total/1e6:.1f}M)")
    print(f"  Trainable:    {trainable:,} ({trainable/1e6:.3f}M)")
    print(f"  Frozen:       {frozen:,} ({frozen/1e6:.1f}M)")


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("3D Models Test (Med3D only, ViT requires full project)")
    print("=" * 60)

    # Test Med3D ResNet series
    for name in ['med3d_resnet18', 'med3d_resnet34', 'med3d_resnet50']:
        try:
            model = create_model_3d(name, num_classes=2, freeze_backbone=True)
            # Test forward pass
            x = torch.randn(2, 1, 64, 64, 64).to(DEVICE)
            with torch.no_grad():
                out = model(x)
            print(f"  Output shape: {out.shape}")
            print(f"  ✓ {name} OK!")
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")

    print("\n" + "=" * 60)