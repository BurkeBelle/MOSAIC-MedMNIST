# preprocess_mosmed.py
"""
MosMedData Preprocessing Script
Convert NIfTI CT volumes to 64x64x64 numpy arrays for 3D external validation.

Task: Binary classification - CT-0 (Normal) vs CT-1~4 (COVID-19 positive)
Input: /root/lanyun-tmp/MosMedData/CT-{0,1,2,3,4}/*.nii
Output: /root/lanyun-tmp/MosMedData/processed_64/
    - train.npy, train_labels.npy
    - test.npy, test_labels.npy
    - data_split.json

No GPU required. Pure CPU operation.
"""

import os
import glob
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

# ============ Configuration ============
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_MosMedData")
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed_64")
TARGET_SIZE = (64, 64, 64)
RANDOM_SEED = 42
TEST_RATIO = 0.2  # 80% train, 20% test

# HU value clipping for lung CT
HU_MIN = -1000  # air
HU_MAX = 400    # soft tissue / bone boundary

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_volume(nii_path, target_size=(64, 64, 64)):
    """Load a NIfTI file and resize to target_size."""
    # Load NIfTI
    img = nib.load(nii_path)
    volume = img.get_fdata().astype(np.float32)
    
    # MosMedData note: during DICOM-to-NIfTI conversion, 
    # only every 10th slice was preserved, so z-axis is already sparse.
    
    # Clip HU values
    volume = np.clip(volume, HU_MIN, HU_MAX)
    
    # Normalize to [0, 1]
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    
    # Resize to target size
    current_size = volume.shape
    zoom_factors = [t / c for t, c in zip(target_size, current_size)]
    volume_resized = zoom(volume, zoom_factors, order=1)  # bilinear interpolation
    
    return volume_resized


def main():
    print("=" * 60)
    print("MosMedData Preprocessing")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Collect all files with labels
    # Label 0 = Normal (CT-0), Label 1 = COVID (CT-1,2,3,4)
    all_files = []
    all_labels = []
    
    for ct_category in ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]:
        ct_dir = os.path.join(DATA_ROOT, ct_category)
        if not os.path.exists(ct_dir):
            print(f"WARNING: {ct_dir} not found, skipping.")
            continue
        
        # Try .nii files first
        nii_files = sorted(glob.glob(os.path.join(ct_dir, "*.nii")))
        # Exclude .nii.gz
        nii_files = [f for f in nii_files if not f.endswith('.nii.gz')]
        
        # If no .nii found, try .nii.gz
        if not nii_files:
            nii_files = sorted(glob.glob(os.path.join(ct_dir, "*.nii.gz")))
        
        label = 0 if ct_category == "CT-0" else 1
        
        print(f"  {ct_category}: {len(nii_files)} files -> label {label}")
        all_files.extend(nii_files)
        all_labels.extend([label] * len(nii_files))
    
    print(f"\nTotal: {len(all_files)} files")
    print(f"  Normal (label 0): {all_labels.count(0)}")
    print(f"  COVID  (label 1): {all_labels.count(1)}")
    
    if len(all_files) == 0:
        print("ERROR: No NIfTI files found!")
        return
    
    # Train/test split (stratified)
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=all_labels
    )
    
    print(f"\nTrain: {len(train_files)} (Normal: {train_labels.count(0)}, COVID: {train_labels.count(1)})")
    print(f"Test:  {len(test_files)} (Normal: {test_labels.count(0)}, COVID: {test_labels.count(1)})")
    
    # Save data split info
    split_info = {
        "train_files": [os.path.basename(f) for f in train_files],
        "train_labels": train_labels,
        "test_files": [os.path.basename(f) for f in test_files],
        "test_labels": test_labels,
        "label_mapping": {"0": "Normal (CT-0)", "1": "COVID-19 (CT-1~4)"},
        "random_seed": RANDOM_SEED,
        "test_ratio": TEST_RATIO,
    }
    with open(os.path.join(OUTPUT_DIR, "data_split.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSaved data split to {os.path.join(OUTPUT_DIR, 'data_split.json')}")
    
    # Process train set
    print(f"\n--- Processing train set ({len(train_files)} volumes) ---")
    train_volumes = []
    train_valid_labels = []
    for i, (fpath, label) in enumerate(zip(train_files, train_labels)):
        try:
            vol = preprocess_volume(fpath, TARGET_SIZE)
            train_volumes.append(vol)
            train_valid_labels.append(label)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(train_files)}] {os.path.basename(fpath)} "
                      f"shape={vol.shape} range=[{vol.min():.3f}, {vol.max():.3f}]")
        except Exception as e:
            print(f"  ERROR processing {fpath}: {e}")
    
    train_arr = np.array(train_volumes, dtype=np.float32)
    train_lab = np.array(train_valid_labels, dtype=np.int64)
    
    np.save(os.path.join(OUTPUT_DIR, "train.npy"), train_arr)
    np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), train_lab)
    print(f"  Saved train.npy: shape={train_arr.shape}, dtype={train_arr.dtype}")
    print(f"  Saved train_labels.npy: shape={train_lab.shape}")
    
    # Free memory
    del train_volumes, train_arr
    
    # Process test set
    print(f"\n--- Processing test set ({len(test_files)} volumes) ---")
    test_volumes = []
    test_valid_labels = []
    for i, (fpath, label) in enumerate(zip(test_files, test_labels)):
        try:
            vol = preprocess_volume(fpath, TARGET_SIZE)
            test_volumes.append(vol)
            test_valid_labels.append(label)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(test_files)}] {os.path.basename(fpath)} "
                      f"shape={vol.shape} range=[{vol.min():.3f}, {vol.max():.3f}]")
        except Exception as e:
            print(f"  ERROR processing {fpath}: {e}")
    
    test_arr = np.array(test_volumes, dtype=np.float32)
    test_lab = np.array(test_valid_labels, dtype=np.int64)
    
    np.save(os.path.join(OUTPUT_DIR, "test.npy"), test_arr)
    np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), test_lab)
    print(f"  Saved test.npy: shape={test_arr.shape}, dtype={test_arr.dtype}")
    print(f"  Saved test_labels.npy: shape={test_lab.shape}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DONE! Output files:")
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {fname}: {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
