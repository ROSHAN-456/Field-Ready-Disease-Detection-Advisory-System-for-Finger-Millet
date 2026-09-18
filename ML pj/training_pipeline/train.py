"""
train.py — Two-phase training: frozen backbone → full fine-tuning.

Usage:
    python train.py                          # uses config.yaml
    python train.py --config my_config.yaml  # custom config
"""

import os
import argparse
import yaml
import tensorflow as tf
from tensorflow import keras

from data_loader import load_config, build_dataset, build_label_maps, compute_class_weights
from model import build_model, compile_model, unfreeze_backbone, print_model_summary


def main():
    parser = argparse.ArgumentParser(description="Train Millet Doctor multi-head model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Paths ──
    dataset_dir = config["data"]["dataset_dir"]
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    checkpoint_dir = config["training"]["checkpoint"]["save_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    disease_classes, _ = build_label_maps(config)
    class_weights = compute_class_weights(train_dir, disease_classes)
    print(f"\n[WEIGHTS] Class weights (disease head): {len(class_weights)} classes computed")

    print("\n[DATA] Loading datasets...")
    train_ds, n_train = build_dataset(train_dir, config, is_training=True, disease_class_weights=class_weights)
    val_ds, n_val = build_dataset(val_dir, config, is_training=False, disease_class_weights=None)
    print(f"  Train: {n_train} | Val: {n_val}")

    # ── Model ──
    print("\n[MODEL] Building model...")
    model = build_model(config)
    compile_model(model, config)
    print_model_summary(model)

    # ── Callbacks ──
    es_cfg = config["training"]["early_stopping"]
    ck_cfg = config["training"]["checkpoint"]

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=es_cfg["patience"],
            restore_best_weights=es_cfg["restore_best_weights"],
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor=ck_cfg["monitor"],
            save_best_only=ck_cfg["save_best_only"],
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, "training_log.csv"),
            append=False,
        ),
    ]

    # ── Phase 1: Train with frozen backbone ──
    freeze_epochs = config["model"]["freeze_backbone_epochs"]
    total_epochs = config["training"]["epochs"]

    print(f"\n[PHASE 1] Training heads only (backbone frozen) for {freeze_epochs} epochs...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=freeze_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Phase 2: Unfreeze backbone, fine-tune end-to-end ──
    print(f"\n[PHASE 2] Fine-tuning entire model for up to {total_epochs - freeze_epochs} more epochs...")
    unfreeze_backbone(model, config)

    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=freeze_epochs,
        epochs=total_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save final model ──
    saved_model_dir = config["export"]["saved_model_dir"]
    os.makedirs(saved_model_dir, exist_ok=True)
    model.save(os.path.join(saved_model_dir, "final_model.keras"))
    print(f"\n[SAVED] Final model saved to {saved_model_dir}/final_model.keras")

    print("\n[DONE] Training complete! Run evaluate.py to check metrics.")


def export_tflite(model: keras.Model, config: dict, representative_ds):
    """Export the model to quantized TFLite format."""
    saved_model_dir = config["export"]["saved_model_dir"]
    tflite_path = config["export"]["tflite_output"]
    do_quantize = config["export"]["quantize"]

    # Save as SavedModel first (required for TFLite converter)
    sm_path = os.path.join(saved_model_dir, "saved_model")
    model.export(sm_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)

    if do_quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for full INT8 quantization
        def representative_data_gen():
            for batch_images, batch_labels, batch_weights in representative_ds.take(50):
                for i in range(min(batch_images.shape[0], 10)):
                    yield [tf.expand_dims(batch_images[i], 0)]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    target = config["export"]["target_size_mb"]
    status = "✓ PASS" if size_mb < target else "✗ FAIL"
    print(f"  TFLite model: {tflite_path}")
    print(f"  Size: {size_mb:.2f} MB (target <{target} MB) → {status}")


if __name__ == "__main__":
    main()
