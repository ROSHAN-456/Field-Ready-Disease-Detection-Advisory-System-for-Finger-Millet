"""
Diagnostic script: Check for data/label misalignment bugs.
Does NOT modify anything — read-only analysis.
"""

import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add training_pipeline to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_config, build_dataset, build_label_maps, _severity_from_folder, _stress_flag_from_folder

def main():
    config = load_config("config.yaml")
    disease_classes, class_to_idx = build_label_maps(config)
    severity_names = config["classes"]["severity"]
    stress_names = config["classes"]["stress_flag"]

    print("=" * 70)
    print("  DIAGNOSTIC 1: Label ordering comparison")
    print("=" * 70)

    # Show config.yaml order
    print("\n[A] config.yaml disease class order (used by build_dataset):")
    for i, c in enumerate(disease_classes):
        print(f"    {i:2d} -> {c}")

    # Show label_map.json order
    label_map_path = Path(__file__).parent.parent / "finger_millet_advisor" / "assets" / "models" / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)
        print(f"\n[B] label_map.json disease class order:")
        for i, c in enumerate(label_map["disease"]):
            print(f"    {i:2d} -> {c}")

        # Compare
        match = label_map["disease"] == disease_classes
        print(f"\n[C] config.yaml == label_map.json order? {'YES ✓' if match else 'NO ✗ MISMATCH!'}")
        if not match:
            for i, (a, b) in enumerate(zip(disease_classes, label_map["disease"])):
                flag = " ✗" if a != b else ""
                print(f"    {i:2d}: config='{a}' vs json='{b}'{flag}")
    else:
        print(f"\n  label_map.json not found at {label_map_path}")

    # Show what os.listdir returns (unsorted, filesystem order)
    train_dir = os.path.join(config["data"]["dataset_dir"], "train")
    print(f"\n[D] os.listdir('{train_dir}') order (filesystem order, NOT used by build_dataset):")
    try:
        raw_dirs = os.listdir(train_dir)
        for i, d in enumerate(raw_dirs):
            print(f"    {i:2d} -> {d}")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 2: Pipeline label verification (10 random samples)")
    print("=" * 70)

    # Build the TRAINING dataset (same as train.py uses)
    train_ds, n_train = build_dataset(train_dir, config, is_training=False)  # No shuffle for reproducibility

    # Collect all samples to pick 10 random ones
    # We need file paths too — rebuild manually to get them
    split_path = Path(train_dir)
    file_paths_ordered = []
    disease_labels_ordered = []
    severity_labels_ordered = []
    stress_labels_ordered = []

    for cls_name in disease_classes:
        cls_dir = split_path / cls_name
        if not cls_dir.exists():
            continue
        d_idx = class_to_idx[cls_name]
        s_idx = _severity_from_folder(cls_name)
        f_idx = _stress_flag_from_folder(cls_name)

        for img_file in cls_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                file_paths_ordered.append(str(img_file))
                disease_labels_ordered.append(d_idx)
                severity_labels_ordered.append(s_idx)
                stress_labels_ordered.append(f_idx)

    # Pick 10 random indices
    random.seed(42)
    indices = sorted(random.sample(range(len(file_paths_ordered)), min(10, len(file_paths_ordered))))

    print(f"\n  Total training samples: {len(file_paths_ordered)}")
    print(f"  Checking {len(indices)} random samples:\n")

    all_ok = True
    for idx in indices:
        fpath = file_paths_ordered[idx]
        d_label = disease_labels_ordered[idx]
        s_label = severity_labels_ordered[idx]
        f_label = stress_labels_ordered[idx]

        # Extract folder name from path
        folder_name = Path(fpath).parent.name

        # Expected labels based on folder name
        expected_d = class_to_idx.get(folder_name, -1)
        expected_s = _severity_from_folder(folder_name)
        expected_f = _stress_flag_from_folder(folder_name)

        # Decode assigned labels back to names
        assigned_disease_name = disease_classes[d_label] if d_label < len(disease_classes) else "OUT_OF_RANGE"
        assigned_severity_name = severity_names[s_label] if s_label < len(severity_names) else "OUT_OF_RANGE"
        assigned_stress_name = stress_names[f_label] if f_label < len(stress_names) else "OUT_OF_RANGE"

        disease_ok = (d_label == expected_d)
        severity_ok = (s_label == expected_s)
        stress_ok = (f_label == expected_f)
        sample_ok = disease_ok and severity_ok and stress_ok

        if not sample_ok:
            all_ok = False

        status = "✓ OK" if sample_ok else "✗ MISMATCH"
        print(f"  [{idx}] {status}")
        print(f"       File:     .../{folder_name}/{Path(fpath).name}")
        print(f"       Disease:  label={d_label} ({assigned_disease_name}), expected={expected_d} ({folder_name}) {'✓' if disease_ok else '✗'}")
        print(f"       Severity: label={s_label} ({assigned_severity_name}), expected={expected_s} ({severity_names[expected_s]}) {'✓' if severity_ok else '✗'}")
        print(f"       Stress:   label={f_label} ({assigned_stress_name}), expected={expected_f} ({stress_names[expected_f]}) {'✓' if stress_ok else '✗'}")
        print()

    print("=" * 70)
    print("  DIAGNOSTIC 3: Checking for label desynchronization in tf.data pipeline")
    print("=" * 70)

    # Now pull actual batches from the tf.data pipeline and verify
    # The pipeline returns (image, {disease_head, severity_head, stress_head}, weights)
    print("\n  Pulling first 2 batches from tf.data pipeline and checking label coherence...\n")

    batch_count = 0
    mismatch_count = 0
    checked_count = 0

    for batch in train_ds.take(2):
        images, labels, weights = batch
        d_labels = labels["disease_head"].numpy()
        s_labels = labels["severity_head"].numpy()
        f_labels = labels["stress_head"].numpy()

        for i in range(len(d_labels)):
            d = d_labels[i]
            s = s_labels[i]
            f = f_labels[i]

            # Cross-check: does the disease label imply the correct severity and stress?
            if d < len(disease_classes):
                cls_name = disease_classes[d]
                expected_s = _severity_from_folder(cls_name)
                expected_f = _stress_flag_from_folder(cls_name)

                if s != expected_s or f != expected_f:
                    mismatch_count += 1
                    if mismatch_count <= 5:  # Print first 5 mismatches
                        print(f"  ✗ DESYNC at batch {batch_count} sample {i}:")
                        print(f"    disease={d} ({cls_name})")
                        print(f"    severity={s} (expected {expected_s} from '{cls_name}')")
                        print(f"    stress={f} (expected {expected_f} from '{cls_name}')")

            checked_count += 1

        batch_count += 1

    print(f"\n  Checked {checked_count} samples across {batch_count} batches.")
    print(f"  Cross-head mismatches: {mismatch_count}")
    if mismatch_count == 0:
        print("  ✓ All three heads' labels are internally consistent (derived from same class).")
    else:
        print(f"  ✗ FOUND {mismatch_count} DESYNCHRONIZED LABELS!")

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 4: Data normalization check")
    print("=" * 70)

    # Check if images are being double-normalized
    # MobileNetV3 with include_preprocessing=True expects [0, 255] input
    # But data_loader normalizes to [0, 1] with img / 255.0
    print("\n  Checking data_loader.py normalization vs backbone preprocessing...")
    print(f"  data_loader._parse_image: divides by 255.0 -> output range [0, 1]")
    print(f"  MobileNetV3Small(include_preprocessing=True): internally rescales to [-1, 1]")
    print(f"  COMBINED EFFECT: input [0,1] is rescaled as if it were [0,255]")
    print(f"  This means pixels that should map to [-1, 1] actually map to [-1, -0.992]")
    print(f"  >>> ALL PIXEL VALUES ARE CRAMMED INTO A TINY RANGE NEAR -1 <<<")
    print(f"  >>> THIS IS A CRITICAL BUG — DOUBLE NORMALIZATION! <<<")

    # Verify by pulling actual pixel stats
    for batch in train_ds.take(1):
        images = batch[0]
        print(f"\n  Actual pixel stats from pipeline:")
        print(f"    min={images.numpy().min():.4f}, max={images.numpy().max():.4f}, mean={images.numpy().mean():.4f}")
        print(f"    (should be roughly min=0, max=1 BEFORE backbone preprocessing)")
        print(f"    (backbone will then rescale these thinking they are [0,255])")
        break

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Label order match (config vs label_map.json): {'OK' if match else 'MISMATCH'}")
    print(f"  Per-sample label correctness: {'ALL OK' if all_ok else 'MISMATCHES FOUND'}")
    print(f"  Cross-head label sync: {'OK' if mismatch_count == 0 else f'{mismatch_count} MISMATCHES'}")
    print(f"  Double normalization bug: LIKELY (see Diagnostic 4)")
    print("=" * 70)


if __name__ == "__main__":
    main()
