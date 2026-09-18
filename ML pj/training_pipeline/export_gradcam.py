import os
import tensorflow as tf

def build_gradcam_model(model_path):
    print(f"Loading {model_path}...")
    original_model = tf.keras.models.load_model(model_path)
    
    backbone = None
    for l in original_model.layers:
        if isinstance(l, tf.keras.Model):
            backbone = l
            break
            
    if backbone is None:
        raise ValueError("Backbone model not found in layers.")
        
    part1 = backbone
    
    # Extract weights
    # bn_shared
    bn = original_model.get_layer('bn_shared')
    gamma = bn.gamma
    var = bn.moving_variance
    eps = bn.epsilon
    bn_scale = gamma / tf.sqrt(var + eps) # (576,)
    
    # shared_embedding
    se = original_model.get_layer('shared_embedding')
    W1 = se.kernel # (576, 256)
    b1 = se.bias
    
    # disease_dense
    dd = original_model.get_layer('disease_dense')
    W2 = dd.kernel # (256, 128)
    b2 = dd.bias
    
    # disease_head
    dh = original_model.get_layer('disease_head')
    W3 = dh.kernel # (128, 16)
    # we don't need bias for gradient
    
    class GradCamModel(tf.keras.Model):
        def __init__(self, p1):
            super().__init__()
            self.part1 = p1
            # Assign weights as variables/constants so they are saved
            self.bn_scale = tf.constant(bn_scale)
            self.W1 = tf.constant(W1)
            self.b1 = tf.constant(b1)
            self.W2 = tf.constant(W2)
            self.b2 = tf.constant(b2)
            self.W3 = tf.constant(W3)
            
        @tf.function(input_signature=[
            tf.TensorSpec([1, 224, 224, 3], tf.float32, name='input_image'),
            tf.TensorSpec([], tf.int32, name='class_index')
        ])
        def call(self, input_image, class_index):
            # Forward pass to get activations
            features = self.part1(input_image) # [1, H, W, 576]
            H = tf.shape(features)[1]
            W_spatial = tf.shape(features)[2]
            
            x = tf.reduce_mean(features, axis=[1, 2]) # global_pool
            x = x * self.bn_scale # simplified BN forward (only scale matters for gradient)
            
            x1 = tf.matmul(x, self.W1) + self.b1
            x1_relu = tf.nn.relu(x1)
            
            x2 = tf.matmul(x1_relu, self.W2) + self.b2
            x2_relu = tf.nn.relu(x2)
            
            # We don't need to compute the final softmax, just the gradient w.r.t the logit
            # logit = matmul(x2_relu, W3)[..., class_index]
            
            # ── Backward Pass (Chain Rule) ──
            # d_logit / d_x2_relu
            # Select the class_index column from W3
            # W3 is [128, 16]
            dL_dx2_relu = self.W3[:, class_index] # (128,)
            dL_dx2_relu = tf.expand_dims(dL_dx2_relu, 0) # [1, 128]
            
            # d_x2_relu / d_x2 is the step function (1 if x2 > 0 else 0)
            dL_dx2 = dL_dx2_relu * tf.cast(x2 > 0, tf.float32) # [1, 128]
            
            # dL / d_x1_relu
            dL_dx1_relu = tf.matmul(dL_dx2, self.W2, transpose_b=True) # [1, 256]
            
            # dL / d_x1
            dL_dx1 = dL_dx1_relu * tf.cast(x1 > 0, tf.float32) # [1, 256]
            
            # dL / d_x (input to shared_embedding)
            dL_dx = tf.matmul(dL_dx1, self.W1, transpose_b=True) # [1, 576]
            
            # dL / d_global_pool (before BN)
            dL_dpool = dL_dx * self.bn_scale # [1, 576]
            
            # dL / d_features
            # Global average pooling gradient is just broadcasting and dividing by H*W
            # But wait, Grad-CAM uses global average pooling of the *gradients* themselves!
            # pooled_grads = tf.reduce_mean(dL_dfeatures, axis=(1, 2))
            # Since dL_dfeatures is dL_dpool / (H*W) everywhere spatially, 
            # its spatial mean is EXACTLY dL_dpool / (H*W) !
            pooled_grads = dL_dpool / tf.cast(H * W_spatial, tf.float32) # [1, 576]
            
            # Grad-CAM heatmap
            # cam = sum(features * pooled_grads)
            # features: [1, H, W, 576]
            # pooled_grads: [1, 576] -> broadcast to [1, 1, 1, 576]
            pooled_grads_expanded = tf.reshape(pooled_grads, [1, 1, 1, -1])
            cam = tf.reduce_sum(features * pooled_grads_expanded, axis=-1) # [1, H, W]
            cam = tf.nn.relu(cam) # [1, H, W]
            
            # Normalize to [0, 1]
            cam_min = tf.reduce_min(cam)
            cam_max = tf.reduce_max(cam)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            return tf.expand_dims(cam, axis=-1) # [1, H, W, 1]

    return GradCamModel(part1)

def main():
    keras_path = 'exported_model/final_model.keras'
    tflite_path = '../finger_millet_advisor/assets/models/gradcam_model.tflite'
    
    gradcam_model = build_gradcam_model(keras_path)
    
    print("Converting to TFLite...")
    run_model = tf.function(lambda x, y: gradcam_model(x, y))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 224, 224, 3], tf.float32),
        tf.TensorSpec([], tf.int32)
    )
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # We do NOT need SELECT_TF_OPS anymore!
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Saved Grad-CAM TFLite model to {tflite_path}")
    print(f"Size: {len(tflite_model) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
