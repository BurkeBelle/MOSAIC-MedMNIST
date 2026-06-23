"""
Test UniMiSS model and weight loading
"""
import torch
import sys
sys.path.insert(0, '.')

from models_unimiss import UniMiSSClassifier, create_unimiss_classifier

def test_model_structure():
    """Test model structure without pretrained weights"""
    print("=" * 60)
    print("Testing model structure...")
    print("=" * 60)
    
    model = UniMiSSClassifier(num_classes=2, in_chans_2d=3, in_chans_3d=1)
    
    # Print model structure
    print("\nModel structure:")
    for name, module in model.transformer.named_children():
        if hasattr(module, '__len__'):
            print(f"  {name}: {len(module)} modules")
        else:
            print(f"  {name}: {type(module).__name__}")
    
    # Test 2D forward
    print("\n2D forward test...")
    x_2d = torch.randn(2, 3, 224, 224)
    out_2d = model(x_2d)
    print(f"  Input: {x_2d.shape} -> Output: {out_2d.shape}")
    
    # Test 3D forward
    print("\n3D forward test...")
    x_3d = torch.randn(2, 1, 64, 64, 64)
    out_3d = model(x_3d)
    print(f"  Input: {x_3d.shape} -> Output: {out_3d.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    
    return True


def test_weight_loading(weight_path):
    """Test weight loading from checkpoint"""
    print("\n" + "=" * 60)
    print(f"Testing weight loading from: {weight_path}")
    print("=" * 60)
    
    # Load checkpoint and check structure
    checkpoint = torch.load(weight_path, map_location='cpu')
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    if 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
        print(f"Using 'teacher' weights, {len(state_dict)} keys")
    
    # Count encoder keys
    transformer_keys = [k for k in state_dict.keys() if k.startswith('backbone.transformer.')]
    print(f"Transformer keys in checkpoint: {len(transformer_keys)}")
    
    # Create model and load weights
    model = create_unimiss_classifier(
        num_classes=2, 
        pretrained_path=weight_path,
        in_chans_2d=3,
        in_chans_3d=1
    )
    
    # Check which keys were loaded
    model_keys = set(model.transformer.state_dict().keys())
    loaded_keys = set(k.replace('backbone.transformer.', '') for k in transformer_keys)
    
    matched = model_keys & loaded_keys
    missing_in_ckpt = model_keys - loaded_keys
    extra_in_ckpt = loaded_keys - model_keys
    
    print(f"\nKey matching:")
    print(f"  Model keys: {len(model_keys)}")
    print(f"  Checkpoint transformer keys: {len(loaded_keys)}")
    print(f"  Matched: {len(matched)}")
    print(f"  Missing in checkpoint: {len(missing_in_ckpt)}")
    print(f"  Extra in checkpoint: {len(extra_in_ckpt)}")
    
    if missing_in_ckpt:
        print(f"\n  Sample missing keys: {list(missing_in_ckpt)[:5]}")
    if extra_in_ckpt:
        print(f"\n  Sample extra keys: {list(extra_in_ckpt)[:5]}")
    
    # Test forward after loading
    print("\nForward test after loading weights...")
    model.eval()
    with torch.no_grad():
        x_2d = torch.randn(1, 3, 224, 224)
        out_2d = model(x_2d)
        print(f"  2D: {x_2d.shape} -> {out_2d.shape}, values: {out_2d[0].tolist()}")
        
        x_3d = torch.randn(1, 1, 64, 64, 64)
        out_3d = model(x_3d)
        print(f"  3D: {x_3d.shape} -> {out_3d.shape}, values: {out_3d[0].tolist()}")
    
    return True


if __name__ == "__main__":
    # Test structure
    test_model_structure()
    
    # Test weight loading if path provided
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to UniMiSS weights')
    args = parser.parse_args()
    
    if args.weights:
        test_weight_loading(args.weights)
    else:
        print("\nNo weights provided. Use --weights to test weight loading.")
        print("Example: python test_unimiss.py --weights weights/self_supervised_unimiss_nnunet_small_5022.pth")