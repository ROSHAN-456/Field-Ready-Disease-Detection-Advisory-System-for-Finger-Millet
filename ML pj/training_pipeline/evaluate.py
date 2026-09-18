"""
evaluate.py — Per-class precision/recall/F1 and confusion matrices for each head.

Usage:
    python evaluate.py                                    # defaults
    python evaluate.py --model ./checkpoints/best_model.keras --split test
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from data_loader import load_config, build_dataset, build_label_maps


def predict_all(model, dataset):
    """Run inference on entire dataset, collecting predictions + ground truth."""
    all_preds = {"disease_head": [], "severity_head": [], "stress_head": []}
    all_labels = {"disease_head": [], "severity_head": [], "stress_head": []}

    for batch in dataset:
        images, labels = batch[0], batch[1]
        preds = model.predict(images, verbose=0)
        for head in all_preds:
            all_preds[head].append(np.argmax(preds[head], axis=-1))
            all_labels[head].append(labels[head].numpy())

    for head in all_preds:
        all_preds[head] = np.concatenate(all_preds[head])
        all_labels[head] = np.concatenate(all_labels[head])

    return all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(class_names)), max(8, len(class_names) * 0.5)))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title} — Normalized")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Millet Doctor model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="./checkpoints/best_model.keras")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)

    # ── Load data ──
    split_dir = os.path.join(config["data"]["dataset_dir"], args.split)
    print(f"Loading {args.split} split from {split_dir}...")
    ds, n_samples = build_dataset(split_dir, config, is_training=False)

    # ── Predict ──
    print(f"Running inference on {n_samples} images...")
    all_preds, all_labels = predict_all(model, ds)

    # ── Evaluate each head ──
    heads = {
        "disease_head": config["classes"]["disease"],
        "severity_head": config["classes"]["severity"],
        "stress_head": config["classes"]["stress_flag"],
    }

    report_lines = [f"# Evaluation Report — {args.split} set",
                    f"**Model:** `{args.model}`",
                    f"**Samples:** {n_samples}\n"]

    for head_name, class_names in heads.items():
        print(f"\n{'='*60}")
        print(f"  HEAD: {head_name}")
        print(f"{'='*60}")

        y_true = all_labels[head_name]
        y_pred = all_preds[head_name]

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            labels=range(len(class_names)),
            zero_division=0,
            output_dict=False,
        )
        print(report)

        report_lines.append(f"## {head_name}")
        report_lines.append(f"```\n{report}\n```\n")

        # Confusion matrix
        cm_path = os.path.join(args.output_dir, f"cm_{head_name}.png")
        plot_confusion_matrix(y_true, y_pred, class_names, head_name, cm_path)

    # ── Save text report ──
    report_path = os.path.join(args.output_dir, f"eval_report_{args.split}.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n[REPORT] Saved: {report_path}")


if __name__ == "__main__":
    main()
