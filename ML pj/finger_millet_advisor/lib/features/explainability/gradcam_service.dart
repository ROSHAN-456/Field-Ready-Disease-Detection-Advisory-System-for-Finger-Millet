/// Explainability module — Grad-CAM heatmap generation.
///
/// Computes class activation maps using the TFLite model's
/// final convolutional layer and overlays them on the leaf image.
library;

import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class GradCamService {
  static Interpreter? _interpreter;

  static Future<void> _init() async {
    if (_interpreter == null) {
      _interpreter = await Interpreter.fromAsset('assets/models/gradcam_model.tflite');
    }
  }

  static Future<Uint8List?> generateHeatmap({
    required String imagePath,
    required int predictedClassIndex,
  }) async {
    try {
      await _init();

      // 1. Load and preprocess image
      final bytes = await File(imagePath).readAsBytes();
      final originalImage = img.decodeImage(bytes);
      if (originalImage == null) return null;

      final resizedImage = img.copyResize(originalImage, width: 224, height: 224);

      var inputBuffer = Float32List(1 * 224 * 224 * 3);
      int pixelIndex = 0;
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resizedImage.getPixel(x, y);
          inputBuffer[pixelIndex++] = pixel.r.toDouble();
          inputBuffer[pixelIndex++] = pixel.g.toDouble();
          inputBuffer[pixelIndex++] = pixel.b.toDouble();
        }
      }

      // Output shape is [1, 7, 7, 1]
      final outputBuffer = List.filled(1 * 7 * 7 * 1, 0.0).reshape([1, 7, 7, 1]);

      // 2. Run interpreter
      // The model signature requires 'input_image' and 'class_index'.
      // We pass class_index as a scalar (just the int).
      _interpreter!.runForMultipleInputs(
        [inputBuffer.reshape([1, 224, 224, 3]), predictedClassIndex],
        {0: outputBuffer},
      );

      // 3. Process heatmap
      final heatmapData = (outputBuffer as List)[0] as List; // 7x7x1
      final w = 7;
      final h = 7;
      
      final heatmapImg = img.Image(width: w, height: h);
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          double val = (heatmapData[y][x][0] as double);
          final color = _valueToJet(val);
          heatmapImg.setPixel(x, y, color);
        }
      }

      // Upscale heatmap to original image size
      final upscaledHeatmap = img.copyResize(heatmapImg, width: originalImage.width, height: originalImage.height, interpolation: img.Interpolation.linear);

      // Blend
      for (int y = 0; y < originalImage.height; y++) {
        for (int x = 0; x < originalImage.width; x++) {
          final origPx = originalImage.getPixel(x, y);
          final heatPx = upscaledHeatmap.getPixel(x, y);
          
          final blendedR = ((origPx.r * 0.6) + (heatPx.r * 0.4)).toInt().clamp(0, 255);
          final blendedG = ((origPx.g * 0.6) + (heatPx.g * 0.4)).toInt().clamp(0, 255);
          final blendedB = ((origPx.b * 0.6) + (heatPx.b * 0.4)).toInt().clamp(0, 255);
          
          originalImage.setPixelRgba(x, y, blendedR, blendedG, blendedB, 255);
        }
      }

      return img.encodeJpg(originalImage);
    } catch (e) {
      print('GradCam error: $e');
      return null;
    }
  }

  static img.Color _valueToJet(double v) {
    double r = 0, g = 0, b = 0;
    if (v < 0.25) {
      b = 1.0;
      g = 4.0 * v;
    } else if (v < 0.5) {
      b = 1.0 - 4.0 * (v - 0.25);
      g = 1.0;
    } else if (v < 0.75) {
      r = 4.0 * (v - 0.5);
      g = 1.0;
    } else {
      r = 1.0;
      g = 1.0 - 4.0 * (v - 0.75);
    }
    return img.ColorRgb8((r * 255).toInt(), (g * 255).toInt(), (b * 255).toInt());
  }
}
