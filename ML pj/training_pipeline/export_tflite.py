"""
export_tflite.py — Export best_model.keras to quantized TFLite.
Uses float16 quantization for better accuracy preservation.
"""
import os
import numpy as np
import tensorflow as tf
from data_loader import load_config, build_dataset

def main():
    config = load_config("config.yaml")
    
    keras_path = "./checkpoints/best_model.keras"
    tflite_path = config["export"]["tflite_output"]
    
    print(f"[1/3] Loading Keras model from {keras_path}...")
    model = tf.keras.models.load_model(keras_path)
    
    # Save as SavedModel (required for TFLite converter)
    sm_path = os.path.join(config["export"]["saved_model_dir"], "saved_model")
    os.makedirs(sm_path, exist_ok=True)
    model.export(sm_path)
    print(f"  Exported SavedModel to {sm_path}")
    
    print("[2/3] Converting to TFLite with FLOAT16 quantization...")
    converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
    
    # Use float16 quantization — much better accuracy than INT8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    # Keep float32 input/output for easy pipeline compatibility
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    target = config["export"]["target_size_mb"]
    status = "PASS" if size_mb < target else "FAIL"
    
    print(f"[3/3] Done!")
    print(f"  TFLite model: {tflite_path}")
    print(f"  Size: {size_mb:.2f} MB (target <{target} MB) -> {status}")

if __name__ == "__main__":
    main()
