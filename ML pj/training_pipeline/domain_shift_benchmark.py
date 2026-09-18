"""
domain_shift_benchmark.py — Compare model performance on clean lab images vs field-condition images.

This is a key deliverable: it proves the model generalizes beyond controlled settings.

Expected directory layout:
    benchmark_data/
    ├── lab/           ← Clean, well-lit, white-background images
    │   ├── blast_mild/
    │   ├── blast_moderate/
    │   └── ...
    └── field/         ← Real farm photos, cluttered backgrounds, varied lighting
        ├── blast_mild/
        ├── blast_moderate/
        └── ...

Usage:
    python domain_shift_benchmark.py
    python domain_shift_benchmark.py --lab_dir ./benchmark_data/lab --field_dir ./benchmark_data/field
"""

import os
import argparse
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras

from data_loader import load_config, build_dataset, build_label_maps


def evaluate_on_split(model, dataset, config, domain_name):
    """Evaluate model on a dataset and return metrics dict."""
    heads = {
        "disease_head": config["classes"]["disease"],
        "severity_head": config["classes"]["severity"],
        "stress_head": config["classes"]["stress_flag"],
    }

    all_preds = {h: [] for h in heads}
    all_labels = {h: [] for h in heads}

    for batch in dataset:
        images, labels = batch[0], batch[1]
        preds = model.predict(images, verbose=0)
        for head in heads:
            all_preds[head].append(np.argmax(preds[head], axis=-1))
            all_labels[head].append(labels[head].numpy())

    for head in heads:
        all_preds[head] = np.concatenate(all_preds[head])
        all_labels[head] = np.concatenate(all_labels[head])

    results = {}
    for head_name, class_names in heads.items():
        y_true = all_labels[head_name]
        y_pred = all_preds[head_name]

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            labels=range(len(class_names)),
            zero_division=0,
            output_dict=True,
        )

        results[head_name] = {
            "accuracy": acc,
            "weighted_f1": report["weighted avg"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {
                cls: {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1": report[cls]["f1-score"],
                    "support": report[cls]["support"],
                }
                for cls in class_names if cls in report
            },
        }

    return results


def generate_comparison_report(lab_results, field_results, output_path):
    """Generate a Markdown comparison report."""
    lines = [
        "# Domain Shift Benchmark Report",
        "",
        "Compares model performance on **lab-condition** images (clean, controlled lighting)",
        "vs **field-condition** images (cluttered backgrounds, phone cameras, farm lighting).",
        "",
        "A significant gap indicates the model has not generalized beyond clean data.",
        "",
    ]

    heads = list(lab_results.keys())

    for head in heads:
        lab = lab_results[head]
        field = field_results[head]

        acc_drop = lab["accuracy"] - field["accuracy"]
        f1_drop = lab["weighted_f1"] - field["weighted_f1"]

        lines.append(f"## {head}")
        lines.append("")
        lines.append("| Metric | Lab | Field | Gap |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Accuracy | {lab['accuracy']:.4f} | {field['accuracy']:.4f} | {acc_drop:+.4f} |")
        lines.append(f"| Weighted F1 | {lab['weighted_f1']:.4f} | {field['weighted_f1']:.4f} | {f1_drop:+.4f} |")
        lines.append(f"| Macro F1 | {lab['macro_f1']:.4f} | {field['macro_f1']:.4f} | {lab['macro_f1'] - field['macro_f1']:+.4f} |")
        lines.append("")

        # Severity assessment
        if abs(acc_drop) < 0.05:
            verdict = "[EXCELLENT] Minimal domain shift (<5% accuracy gap)"
        elif abs(acc_drop) < 0.10:
            verdict = "[ACCEPTABLE] Moderate domain shift (5-10% gap). More field augmentation recommended."
        else:
            verdict = "[CONCERNING] Significant domain shift (>10% gap). Model needs more field-condition training data."

        lines.append(f"**Verdict:** {verdict}")
        lines.append("")

        # Per-class breakdown table
        lab_classes = lab["per_class"]
        field_classes = field["per_class"]
        all_cls = set(lab_classes.keys()) | set(field_classes.keys())

        if all_cls:
            lines.append("### Per-Class F1 Comparison")
            lines.append("| Class | Lab F1 | Field F1 | Gap |")
            lines.append("|---|---|---|---|")
            for cls in sorted(all_cls):
                l_f1 = lab_classes.get(cls, {}).get("f1", 0)
                f_f1 = field_classes.get(cls, {}).get("f1", 0)
                gap = l_f1 - f_f1
                flag = " [WARN]" if gap > 0.15 else ""
                lines.append(f"| {cls} | {l_f1:.3f} | {f_f1:.3f} | {gap:+.3f}{flag} |")
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n[REPORT] Domain shift report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Domain shift benchmark")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="./checkpoints/best_model.keras")
    parser.add_argument("--lab_dir", type=str, default="./benchmark_data/lab")
    parser.add_argument("--field_dir", type=str, default="./benchmark_data/field")
    parser.add_argument("--output", type=str, default="./eval_results/domain_shift_report.md")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ── Load model ──
    print(f"Loading model: {args.model}")
    model = keras.models.load_model(args.model)

    # ── Evaluate on Lab images ──
    print(f"\n[LAB] Evaluating on LAB images: {args.lab_dir}")
    lab_ds, n_lab = build_dataset(args.lab_dir, config, is_training=False)
    print(f"  Lab samples: {n_lab}")
    lab_results = evaluate_on_split(model, lab_ds, config, "lab")

    # ── Evaluate on Field images ──
    print(f"\n[FIELD] Evaluating on FIELD images: {args.field_dir}")
    field_ds, n_field = build_dataset(args.field_dir, config, is_training=False)
    print(f"  Field samples: {n_field}")
    field_results = evaluate_on_split(model, field_ds, config, "field")

    # ── Generate comparison ──
    generate_comparison_report(lab_results, field_results, args.output)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  DOMAIN SHIFT SUMMARY")
    print(f"{'='*60}")
    for head in lab_results:
        lab_acc = lab_results[head]["accuracy"]
        field_acc = field_results[head]["accuracy"]
        gap = lab_acc - field_acc
        print(f"  {head:20s}  Lab: {lab_acc:.3f}  Field: {field_acc:.3f}  Gap: {gap:+.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
