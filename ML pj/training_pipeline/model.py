"""
model.py — Multi-head MobileNetV3-Small / EfficientNet-Lite0 model.

Architecture:
  ┌─────────────────┐
  │  Input (224x224) │
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │  MobileNetV3    │  ← ImageNet pretrained, initially frozen
  │  (backbone)     │
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │  GlobalAvgPool  │
  │  + Dropout      │
  │  + Dense(256)   │  ← Shared embedding
  └────────┬────────┘
       ┌───┼───────────┐
       │   │           │
  ┌────▼┐ ┌▼────┐  ┌──▼───┐
  │Dis. │ │Sev. │  │Stress│   ← 3 independent heads
  │(16) │ │(4)  │  │(3)   │
  └─────┘ └─────┘  └──────┘
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(config: dict) -> keras.Model:
    """Builds the multi-head classification model."""

    img_size = config["data"]["image_size"]
    backbone_name = config["model"]["backbone"]
    dropout_rate = config["model"]["dropout_rate"]
    embedding_dim = config["model"]["embedding_dim"]

    n_disease = len(config["classes"]["disease"])       # 16
    n_severity = len(config["classes"]["severity"])     # 4
    n_stress = len(config["classes"]["stress_flag"])    # 3

    # ── Input ──
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")

    # ── Backbone ──
    if backbone_name == "MobileNetV3Small":
        backbone = keras.applications.MobileNetV3Small(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet",
            include_preprocessing=True,   # Built-in [-1, 1] normalization
        )
    elif backbone_name == "EfficientNetLite0":
        # EfficientNetB0 is the closest TF equivalent to Lite0
        backbone = keras.applications.EfficientNetB0(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    backbone.trainable = False  # Freeze initially for transfer learning

    x = backbone(inputs, training=False)

    # ── Shared embedding ──
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.BatchNormalization(name="bn_shared")(x)
    x = layers.Dropout(dropout_rate, name="dropout_shared")(x)
    shared = layers.Dense(embedding_dim, activation="relu", name="shared_embedding")(x)

    # ── Head 1: Disease type (16-class) ──
    d = layers.Dense(128, activation="relu", name="disease_dense")(shared)
    d = layers.Dropout(dropout_rate / 2, name="disease_dropout")(d)
    disease_out = layers.Dense(n_disease, activation="softmax", name="disease_head")(d)

    # ── Head 2: Severity stage (4-class) ──
    s = layers.Dense(64, activation="relu", name="severity_dense")(shared)
    s = layers.Dropout(dropout_rate / 2, name="severity_dropout")(s)
    severity_out = layers.Dense(n_severity, activation="softmax", name="severity_head")(s)

    # ── Head 3: Stress-vs-disease flag (3-class) ──
    f = layers.Dense(64, activation="relu", name="stress_dense")(shared)
    f = layers.Dropout(dropout_rate / 2, name="stress_dropout")(f)
    stress_out = layers.Dense(n_stress, activation="softmax", name="stress_head")(f)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "disease_head": disease_out,
            "severity_head": severity_out,
            "stress_head": stress_out,
        },
        name="MilletDoctorNet",
    )

    return model


def compile_model(model: keras.Model, config: dict, class_weights: dict = None):
    """Compiles the model with per-head losses and loss weights."""

    lr = config["training"]["initial_lr"]
    label_smoothing = config["training"]["label_smoothing"]
    loss_weights = config["training"]["loss_weights"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            "disease_head": keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
            ),
            "severity_head": keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
            ),
            "stress_head": keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
            ),
        },
        loss_weights=loss_weights,
        metrics={
            "disease_head": ["accuracy"],
            "severity_head": ["accuracy"],
            "stress_head": ["accuracy"],
        },
    )

    return model


def unfreeze_backbone(model: keras.Model, config: dict):
    """Unfreezes the backbone for fine-tuning with a lower learning rate."""

    # Find the backbone layer (it's the first non-input layer)
    backbone_layer = None
    for layer in model.layers:
        if hasattr(layer, 'trainable') and isinstance(layer, keras.Model):
            backbone_layer = layer
            break

    if backbone_layer is None:
        print("WARNING: Could not find backbone layer to unfreeze.")
        return

    backbone_layer.trainable = True

    # Freeze BatchNorm layers in backbone to preserve pretrained stats
    for layer in backbone_layer.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    finetune_lr = config["training"]["finetune_lr"]
    loss_weights = config["training"]["loss_weights"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=finetune_lr),
        loss={
            "disease_head": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "severity_head": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "stress_head": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        },
        loss_weights=loss_weights,
        metrics={
            "disease_head": ["accuracy"],
            "severity_head": ["accuracy"],
            "stress_head": ["accuracy"],
        },
    )

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"Backbone unfrozen. Trainable: {trainable_params:,} / {total_params:,} params")
    print(f"Fine-tune LR: {finetune_lr}")


def print_model_summary(model: keras.Model):
    """Prints model size info relevant to our <10MB TFLite target."""
    total = model.count_params()
    # FP32: 4 bytes/param. INT8 quantized: ~1 byte/param.
    fp32_mb = (total * 4) / (1024 * 1024)
    int8_mb = total / (1024 * 1024)

    print(f"\n{'='*50}")
    print(f"  Model: {model.name}")
    print(f"  Total params:     {total:>12,}")
    print(f"  FP32 size (est):  {fp32_mb:>10.2f} MB")
    print(f"  INT8 size (est):  {int8_mb:>10.2f} MB  <- TFLite target")
    print(f"  Target:           {'PASS' if int8_mb < 10 else 'FAIL (>{10}MB)'}")
    print(f"{'='*50}\n")

    model.summary(expand_nested=False, show_trainable=True)
