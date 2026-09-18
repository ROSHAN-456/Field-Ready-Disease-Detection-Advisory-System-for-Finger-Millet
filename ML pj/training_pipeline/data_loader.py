"""
data_loader.py — Builds tf.data pipelines with multi-head label mapping.

Each image folder name (e.g. "blast_mild") is mapped to THREE labels:
  1. disease_id  — index into the full 16-class list
  2. severity_id — 0=none, 1=mild, 2=moderate, 3=severe
  3. stress_flag — 0=disease, 1=stress, 2=healthy
"""

import os
import yaml
import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import Counter


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _severity_from_folder(folder_name: str) -> int:
    """Derive severity index from folder name suffix."""
    if folder_name.endswith("_mild"):
        return 1
    elif folder_name.endswith("_moderate"):
        return 2
    elif folder_name.endswith("_severe"):
        return 3
    return 0  # healthy or stress → 'none'


def _stress_flag_from_folder(folder_name: str) -> int:
    """Derive stress-vs-disease flag from folder name prefix."""
    if folder_name.startswith("stress_"):
        return 1  # stress
    elif folder_name == "healthy_leaf":
        return 2  # healthy
    return 0      # disease


def build_label_maps(config: dict):
    """Returns the list of disease classes and helper maps."""
    disease_classes = config["classes"]["disease"]
    class_to_idx = {c: i for i, c in enumerate(disease_classes)}
    return disease_classes, class_to_idx


def compute_class_weights(dataset_dir: str, disease_classes: list) -> dict:
    """
    Compute inverse-frequency class weights for the disease head.
    Returns a dict mapping class_index -> weight.
    """
    counts = Counter()
    for cls in disease_classes:
        cls_dir = Path(dataset_dir) / cls
        if cls_dir.exists():
            n = len([f for f in cls_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in ('.jpg', '.png', '.jpeg')])
            counts[cls] = n

    total = sum(counts.values())
    n_classes = len(disease_classes)

    weights = {}
    for i, cls in enumerate(disease_classes):
        c = counts.get(cls, 1)
        # Balanced weighting: total / (n_classes * count_for_class)
        weights[i] = total / (n_classes * c) if c > 0 else 1.0

    return weights


def _parse_image(file_path, label_disease, label_severity, label_stress, img_size):
    """Read, decode, and preprocess a single image."""
    raw = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)  # Keep [0, 255] — backbone has include_preprocessing=True
    return img, {
        "disease_head": label_disease,
        "severity_head": label_severity,
        "stress_head": label_stress,
    }


def build_dataset(split_dir: str, config: dict, is_training: bool = False, disease_class_weights: dict = None):
    """
    Builds a tf.data.Dataset from a split directory (train/val/test).

    Returns:
        dataset: tf.data.Dataset yielding (image, {disease_head, severity_head, stress_head})
        num_samples: total number of images
    """
    img_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]
    prefetch = config["data"]["prefetch_buffer"]

    disease_classes, class_to_idx = build_label_maps(config)

    file_paths = []
    disease_labels = []
    severity_labels = []
    stress_labels = []

    split_path = Path(split_dir)
    for cls_name in disease_classes:
        cls_dir = split_path / cls_name
        if not cls_dir.exists():
            continue

        d_idx = class_to_idx[cls_name]
        s_idx = _severity_from_folder(cls_name)
        f_idx = _stress_flag_from_folder(cls_name)

        for img_file in cls_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                file_paths.append(str(img_file))
                disease_labels.append(d_idx)
                severity_labels.append(s_idx)
                stress_labels.append(f_idx)

    num_samples = len(file_paths)
    if num_samples == 0:
        raise ValueError(f"No images found in {split_dir}")

    print(f"  Loaded {num_samples} images from {split_dir}")


    # Add dummy sample weights for all heads, except disease head uses computed class weights
    if disease_class_weights is None:
        disease_class_weights = {i: 1.0 for i in range(len(disease_classes))}

    # Convert weights to lookup table tensor
    keys = list(disease_class_weights.keys())
    values = list(disease_class_weights.values())
    weight_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value=1.0)

    ds = tf.data.Dataset.from_tensor_slices((
        file_paths,
        disease_labels,
        severity_labels,
        stress_labels,
    ))

    def _mapper(fp, ld, ls, lf):
        img, labels = _parse_image(fp, ld, ls, lf, img_size)
        weights = {
            "disease_head": weight_lookup.lookup(tf.cast(ld, tf.int32)),
            "severity_head": 1.0,
            "stress_head": 1.0,
        }
        return img, labels, weights

    ds = ds.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(buffer_size=min(num_samples, 2048))

    ds = ds.batch(batch_size).prefetch(prefetch)
    return ds, num_samples
