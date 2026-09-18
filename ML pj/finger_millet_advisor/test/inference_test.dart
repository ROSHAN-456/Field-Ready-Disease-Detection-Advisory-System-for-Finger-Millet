import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:finger_millet_advisor/features/inference/inference_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('Run Inference on Dummy Image', () async {
    // We need to provide a dummy image from the dataset
    // D:\ML pj\dataset_pipeline\dataset_split\test\healthy_leaf\...
    
    // Find a test image
    final dir = Directory(r'd:\ML pj\dataset_pipeline\dataset_split\test');
    if (!dir.existsSync()) {
      print("Test dir not found");
      return;
    }
    
    // Get first jpg file
    File? testFile;
    for (var entity in dir.listSync(recursive: true)) {
      if (entity is File && entity.path.endsWith('.jpg')) {
        testFile = entity;
        break;
      }
    }
    
    if (testFile == null) {
      print("No test image found");
      return;
    }
    
    print("Testing with image: ${testFile.path}");
    
    final result = await InferenceService.runDiagnosis(testFile.path);
    
    print("\n=== INFERENCE RESULT ===");
    print("Latency: ${result.inferenceLatencyMs}ms");
    print("Disease: ${result.diseaseName} (${(result.diseaseConfidence * 100).toStringAsFixed(1)}%)");
    print("Severity: ${result.severity} (${(result.severityConfidence * 100).toStringAsFixed(1)}%)");
    print("Stress: ${result.stressFlag} (${(result.stressConfidence * 100).toStringAsFixed(1)}%)");
    print("Is Placeholder: ${result.isPlaceholderModel}");
    
    print("\nRaw Disease Scores:");
    result.rawDiseaseScores.forEach((k, v) => print("  $k: ${v.toStringAsFixed(4)}"));
  });
}
