import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class DiagnosisResult {
  final String diseaseName;
  final double diseaseConfidence;
  final String severity;
  final double severityConfidence;
  final String stressFlag;
  final double stressConfidence;

  final String? stressLookAlike;
  final String advisory;

  final bool isPlaceholderModel;
  final int inferenceLatencyMs;

  final Map<String, double> rawDiseaseScores;
  final Map<String, double> rawSeverityScores;
  final Map<String, double> rawStressScores;

  const DiagnosisResult({
    required this.diseaseName,
    required this.diseaseConfidence,
    required this.severity,
    required this.severityConfidence,
    required this.stressFlag,
    required this.stressConfidence,
    this.stressLookAlike,
    required this.advisory,
    required this.isPlaceholderModel,
    required this.inferenceLatencyMs,
    required this.rawDiseaseScores,
    required this.rawSeverityScores,
    required this.rawStressScores,
  });
}

class InferenceService {
  static Interpreter? _interpreter;
  static Map<String, dynamic>? _labelMap;
  static bool _isInitialized = false;

  static Future<void> init() async {
    if (_isInitialized) return;

    // Load label map
    final labelMapString = await rootBundle.loadString('assets/models/label_map.json');
    _labelMap = json.decode(labelMapString);

    // Load model
    _interpreter = await Interpreter.fromAsset('assets/models/finger_millet_model.tflite');
    _isInitialized = true;
  }

  static Future<DiagnosisResult> runDiagnosis(String imagePath) async {
    await init();
    final startTime = DateTime.now();

    // 1. Preprocess Image
    final bytes = await File(imagePath).readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception("Failed to decode image");

    // Resize to 224x224
    final resizedImage = img.copyResize(image, width: 224, height: 224);

    // Convert to uint8 buffer (RGB)
    var inputBuffer = Uint8List(1 * 224 * 224 * 3);
    int pixelIndex = 0;
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resizedImage.getPixel(x, y);
        inputBuffer[pixelIndex++] = pixel.r.toInt();
        inputBuffer[pixelIndex++] = pixel.g.toInt();
        inputBuffer[pixelIndex++] = pixel.b.toInt();
      }
    }
    
    final inputShape = [1, 224, 224, 3];

    // 2. Prepare outputs
    final outputTensors = _interpreter!.getOutputTensors();
    
    int diseaseIndex = -1;
    int severityIndex = -1;
    int stressIndex = -1;

    for (int i = 0; i < outputTensors.length; i++) {
      final shape = outputTensors[i].shape;
      if (shape.length == 2 && shape[1] == 16) diseaseIndex = i;
      else if (shape.length == 2 && shape[1] == 4) severityIndex = i;
      else if (shape.length == 2 && shape[1] == 3) stressIndex = i;
    }

    if (diseaseIndex == -1 || severityIndex == -1 || stressIndex == -1) {
      throw Exception("Could not map output tensors properly.");
    }

    final outputs = <int, Object>{};
    outputs[diseaseIndex] = List.filled(1 * 16, 0.0).reshape([1, 16]);
    outputs[severityIndex] = List.filled(1 * 4, 0.0).reshape([1, 4]);
    outputs[stressIndex] = List.filled(1 * 3, 0.0).reshape([1, 3]);

    // 3. Run Inference
    _interpreter!.runForMultipleInputs([inputBuffer.reshape(inputShape)], outputs);

    // 4. Parse Results
    final diseaseProbs = (outputs[diseaseIndex] as List)[0] as List<double>;
    final severityProbs = (outputs[severityIndex] as List)[0] as List<double>;
    final stressProbs = (outputs[stressIndex] as List)[0] as List<double>;

    final diseaseLabels = List<String>.from(_labelMap!['disease']);
    final severityLabels = List<String>.from(_labelMap!['severity']);
    final stressLabels = List<String>.from(_labelMap!['stress_flag']);

    Map<String, double> rawDisease = {};
    int bestDiseaseIdx = 0;
    double bestDiseaseProb = -1.0;
    for (int i = 0; i < diseaseLabels.length; i++) {
      rawDisease[diseaseLabels[i]] = diseaseProbs[i];
      if (diseaseProbs[i] > bestDiseaseProb) {
        bestDiseaseProb = diseaseProbs[i];
        bestDiseaseIdx = i;
      }
    }

    Map<String, double> rawSeverity = {};
    int bestSeverityIdx = 0;
    double bestSeverityProb = -1.0;
    for (int i = 0; i < severityLabels.length; i++) {
      rawSeverity[severityLabels[i]] = severityProbs[i];
      if (severityProbs[i] > bestSeverityProb) {
        bestSeverityProb = severityProbs[i];
        bestSeverityIdx = i;
      }
    }

    Map<String, double> rawStress = {};
    int bestStressIdx = 0;
    double bestStressProb = -1.0;
    for (int i = 0; i < stressLabels.length; i++) {
      rawStress[stressLabels[i]] = stressProbs[i];
      if (stressProbs[i] > bestStressProb) {
        bestStressProb = stressProbs[i];
        bestStressIdx = i;
      }
    }

    final latencyMs = DateTime.now().difference(startTime).inMilliseconds;
    final isPlaceholder = _labelMap!['metadata']?['isPlaceholderModel'] ?? true;

    return DiagnosisResult(
      diseaseName: diseaseLabels[bestDiseaseIdx],
      diseaseConfidence: bestDiseaseProb,
      severity: severityLabels[bestSeverityIdx],
      severityConfidence: bestSeverityProb,
      stressFlag: stressLabels[bestStressIdx],
      stressConfidence: bestStressProb,
      stressLookAlike: stressLabels[bestStressIdx] == 'stress' ? 'Looks like environmental stress' : null,
      advisory: 'Advisory pending real data.',
      isPlaceholderModel: isPlaceholder,
      inferenceLatencyMs: latencyMs,
      rawDiseaseScores: rawDisease,
      rawSeverityScores: rawSeverity,
      rawStressScores: rawStress,
    );
  }

  static void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }
}
