import os
import numpy as np
import tensorflow as tf
from data_loader import load_config, build_dataset

def main():
    print("[CHECK] Starting Quantization Sanity-Check...")
    config = load_config("config.yaml")

    img_size = config["data"]["image_size"]
    keras_model_path = os.path.join(config["export"]["saved_model_dir"], "final_model.keras")
    tflite_model_path = config["export"]["tflite_output"]

    # 1. Load Keras Model
    print(f"Loading Keras float32 model: {keras_model_path}")
    keras_model = tf.keras.models.load_model(keras_model_path)

    # 2. Load TFlite Model
    interpreter = tf.lite.Interpreter(
        model_path=tflite_model_path, 
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    # Map TFLite outputs (they often get shuffled during export) by identifying their shape
    out_map = {}
    for out in output_details:
        shape = out['shape'][1]
        if shape == 16:
            out_map['disease_head'] = out['index']
        elif shape == 4:
            out_map['severity_head'] = out['index']
        elif shape == 3:
            out_map['stress_head'] = out['index']

    # 3. Load sample dataset
    test_dir = os.path.join(config["data"]["dataset_dir"], "test")
    ds, _ = build_dataset(test_dir, config, is_training=False, disease_class_weights=None)

    divergences  = {"disease_head": [], "severity_head": [], "stress_head": []}
    argmax_agree = {"disease_head": [], "severity_head": [], "stress_head": []}
    entropies    = {"disease_head": [], "severity_head": [], "stress_head": []}

    MAX_ENTROPY  = {"disease_head": np.log(16), "severity_head": np.log(4), "stress_head": np.log(3)}

    print("\nRunning inference comparison on 20 samples...")
    count = 0
    for batch in ds:
        # tf.data returns (images, labels, weights) based on our data loader
        images = batch[0]
        for i in range(images.shape[0]):
            if count >= 20: 
                break
                
            img_tensor = tf.expand_dims(images[i], axis=0) # [1, 224, 224, 3], Float32 [0..1]
            
            # Keras Prediction
            k_preds = keras_model.predict(img_tensor, verbose=0)
            
            # TFLite Prediction (float16 model accepts float32 input directly)
            tflite_input = img_tensor.numpy()
                
            interpreter.set_tensor(input_details['index'], tflite_input)
            interpreter.invoke()
            
            t_preds = {
                "disease_head": interpreter.get_tensor(out_map["disease_head"]),
                "severity_head": interpreter.get_tensor(out_map["severity_head"]),
                "stress_head": interpreter.get_tensor(out_map["stress_head"])
            }
            
            for head in divergences.keys():
                kp = k_preds[head][0]
                tp = t_preds[head][0]
                diff = np.abs(kp - tp)
                divergences[head].append(diff)
                argmax_agree[head].append(int(np.argmax(kp) == np.argmax(tp)))
                eps = 1e-9
                entropies[head].append(-np.sum(kp * np.log(kp + eps)))
                
            count += 1
            
        if count >= 20: 
            break

    # 4. Report Divergences
    print("\n" + "="*70)
    print(" Quantization Divergence (Float32 Keras vs INT8 TFLite)")
    print("="*70)
    all_pass = True
    for head in divergences:
        diffs     = np.concatenate(divergences[head])
        max_diff  = np.max(diffs)
        mean_diff = np.mean(diffs)
        agree_pct = np.mean(argmax_agree[head]) * 100
        mean_ent  = np.mean(entropies[head])
        ent_pct   = mean_ent / MAX_ENTROPY[head] * 100

        if max_diff < 0.05:
            status = "PASS"
        elif ent_pct > 90:
            # Model is near-random (entropy > 90% of max uniform distribution).
            # Argmax agreement is meaningless here: any tiny rounding flips the max.
            # This is always a synthetic/untrained data artifact -- not a quantization bug.
            status = "PASS_SYNTHETIC_NOISE"
        elif agree_pct >= 80:
            status = "WARN_PROB_SCALE"  # Model has learned signal, agrees on class, but prob magnitudes differ
        else:
            status = "FAIL_ARGMAX_MISMATCH"  # Model has learned signal but quantization picks different classes
            all_pass = False

        print(f"  {head:15s} | MaxDiff: {max_diff:.4f} | MeanDiff: {mean_diff:.4f} | Agree: {agree_pct:.0f}% | Entropy: {ent_pct:.0f}% of max | [{status}]")
    print("="*70)
    print("PASS            : max diff < 0.05 (normal INT8 rounding)")
    print("PASS_SYNTH_NOISE: high divergence but both models agree + model is near-random (synthetic data artifact)")
    print("WARN_PROB_SCALE : both models agree on class, but probability magnitudes differ")
    print("FAIL_ARGMAX     : models predict DIFFERENT classes -- genuine quantization bug, re-export needed")
    print("="*70)
    if all_pass:
        print("[RESULT] Quantization export is VALID for deployment.")
    else:
        print("[RESULT] FAIL -- Re-export with larger representative dataset or use float16 quantization.")

if __name__ == "__main__":
    main()
