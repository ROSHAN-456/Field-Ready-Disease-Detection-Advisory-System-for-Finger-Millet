import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';

import '../advisory/advisory_service.dart';
import '../inference/inference_service.dart';
import '../explainability/gradcam_service.dart';
import '../storage/storage_service.dart';
import '../voice/voice_service.dart';

/// Displays the diagnosis result after a leaf scan.
class ResultScreen extends StatefulWidget {
  final String imagePath;
  final DiagnosisResult diagnosisResult;

  const ResultScreen({
    super.key,
    required this.imagePath,
    required this.diagnosisResult,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  String? _advisoryText;
  bool _advisoryLoading = true;
  bool _isSpeaking = false;
  bool _saved = false;
  String? _saveError;
  bool _gradCamLoading = false;
  Uint8List? _gradCamImageBytes;
  bool _showGradCam = false;
  int? _gradCamLatencyMs;

  DiagnosisResult get r => widget.diagnosisResult;

  @override
  void initState() {
    super.initState();
    _initData();
    _computeGradCam();
  }

  Future<void> _initData() async {
    await _loadAdvisory();
    await _saveDiagnosis();
  }

  Future<void> _computeGradCam() async {
    setState(() => _gradCamLoading = true);
    
    // Find class index from raw scores keys
    final diseaseIdx = r.rawDiseaseScores.keys.toList().indexOf(r.diseaseName);
    
    final startTime = DateTime.now();
    final bytes = await GradCamService.generateHeatmap(
      imagePath: widget.imagePath,
      predictedClassIndex: diseaseIdx,
    );
    final latency = DateTime.now().difference(startTime).inMilliseconds;
    
    if (mounted) {
      setState(() {
        _gradCamImageBytes = bytes;
        _gradCamLoading = false;
        _gradCamLatencyMs = latency;
        // Auto-show it if it successfully generated
        _showGradCam = bytes != null;
      });
    }
  }

  @override
  void dispose() {
    // Stop TTS if navigating away while speaking
    if (_isSpeaking) {
      VoiceService.stop();
    }
    super.dispose();
  }

  Future<void> _loadAdvisory() async {
    try {
      final queryDisease = r.diseaseConfidence < 0.40 ? 'fallback' : r.diseaseName;
      final advisory = await AdvisoryService.getRecommendation(
        diseaseName: queryDisease,
        severity: r.severity,
      );
      if (mounted) {
        setState(() {
          _advisoryText = advisory;
          _advisoryLoading = false;
        });
      }
    } catch (e) {
      debugPrint('Advisory fetch error: $e');
      if (mounted) {
        setState(() {
          _advisoryText = 'Failed to load advisory: $e';
          _advisoryLoading = false;
        });
      }
    }
  }

  Future<void> _saveDiagnosis() async {
    try {
      await StorageService.saveDiagnosis(
        imagePath: widget.imagePath,
        diseaseName: r.diseaseConfidence < 0.40 ? 'Uncertain' : r.diseaseName,
        severity: r.severity,
        confidence: r.diseaseConfidence,
        stressFlag: r.stressFlag,
        stressLookAlike: r.stressLookAlike,
        advisory: _advisoryText ?? r.advisory,
      );
      if (mounted) {
        setState(() => _saved = true);
      }
      debugPrint('Diagnosis saved to SQLite');
    } catch (e) {
      debugPrint('Storage save error: $e');
      if (mounted) {
        setState(() => _saveError = e.toString());
      }
    }
  }

  Future<void> _toggleVoice() async {
    if (_isSpeaking) {
      await VoiceService.stop();
      if (mounted) setState(() => _isSpeaking = false);
    } else {
      final advisoryPart = _advisoryText ?? r.advisory;
      final displayDisease = r.diseaseConfidence < 0.40 ? 'Uncertain' : r.diseaseName;
      final text = 'Disease: $displayDisease. '
          'Severity: ${r.severity}. '
          '$advisoryPart';

      setState(() => _isSpeaking = true);
      await VoiceService.speak(text);
      // TTS fires and forgets — mark as not speaking after call returns
      // (flutter_tts speak() completes immediately; actual playback is async)
      // We'll leave the button as "stop" until the user taps it again.
    }
  }

  Color _severityColor(String severity) {
    switch (severity.toLowerCase()) {
      case 'mild':
        return const Color(0xFF4CAF50);
      case 'moderate':
        return const Color(0xFFFF9800);
      case 'severe':
        return const Color(0xFFF44336);
      default:
        return Colors.grey;
    }
  }

  IconData _severityIcon(String severity) {
    switch (severity.toLowerCase()) {
      case 'mild':
        return Icons.check_circle_rounded;
      case 'moderate':
        return Icons.warning_rounded;
      case 'severe':
        return Icons.dangerous_rounded;
      default:
        return Icons.help_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final sevColor = _severityColor(r.severity);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Diagnosis Result'),
        actions: [
          IconButton(
            onPressed: () {
              // TODO: Share diagnosis
            },
            icon: const Icon(Icons.share_rounded),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (r.isPlaceholderModel)
              Container(
                margin: const EdgeInsets.only(bottom: 16),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.orange.shade100,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.orange.shade300),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.warning_amber_rounded, color: Colors.orange),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        '⚠️ SYNTHETIC MODEL — Results are not medically valid. For testing only.',
                        style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ],
                ),
              ),

            // ── Captured image & Grad-CAM ──
            Stack(
              alignment: Alignment.center,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: _showGradCam && _gradCamImageBytes != null
                        ? Image.memory(
                            _gradCamImageBytes!,
                            fit: BoxFit.cover,
                          )
                        : Image.file(
                            File(widget.imagePath),
                            fit: BoxFit.cover,
                          ),
                  ),
                ),
                if (_gradCamLoading)
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(color: Colors.white),
                        SizedBox(height: 8),
                        Text('Computing Grad-CAM...', style: TextStyle(color: Colors.white)),
                      ],
                    ),
                  ),
                if (_gradCamImageBytes != null)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: Material(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(20),
                      child: InkWell(
                        onTap: () {
                          setState(() {
                            _showGradCam = !_showGradCam;
                          });
                        },
                        borderRadius: BorderRadius.circular(20),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                _showGradCam ? Icons.visibility_off : Icons.visibility,
                                color: Colors.white,
                                size: 16,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                _showGradCam ? 'Hide Heatmap' : 'Show Heatmap',
                                style: const TextStyle(color: Colors.white, fontSize: 12),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 16),

            // ── Disease name + confidence ──
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.biotech_rounded, color: colorScheme.primary),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Disease: ${r.diseaseConfidence < 0.40 ? 'Uncertain' : r.diseaseName}',
                            style: theme.textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    LinearProgressIndicator(
                      value: r.diseaseConfidence,
                      backgroundColor: colorScheme.surfaceContainerHighest,
                      color: colorScheme.primary,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Confidence: ${(r.diseaseConfidence * 100).toStringAsFixed(1)}%',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // ── Severity ──
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(_severityIcon(r.severity), color: sevColor, size: 28),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Severity: ${r.severity.toUpperCase()}',
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              color: sevColor,
                              fontSize: 18,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    LinearProgressIndicator(
                      value: r.severityConfidence,
                      backgroundColor: colorScheme.surfaceContainerHighest,
                      color: sevColor,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Confidence: ${(r.severityConfidence * 100).toStringAsFixed(1)}%',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // ── Stress Flag ──
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.grass_rounded, color: colorScheme.secondary, size: 28),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Classification: ${r.stressFlag.toUpperCase()}',
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              color: colorScheme.secondary,
                              fontSize: 18,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    LinearProgressIndicator(
                      value: r.stressConfidence,
                      backgroundColor: colorScheme.surfaceContainerHighest,
                      color: colorScheme.secondary,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Confidence: ${(r.stressConfidence * 100).toStringAsFixed(1)}%',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // ── Stress look-alike ──
            if (r.stressLookAlike != null)
              Card(
                child: ListTile(
                  leading: Icon(
                    Icons.info_outline_rounded,
                    color: colorScheme.tertiary,
                  ),
                  title: const Text('Possible Stress Look-alike'),
                  subtitle: Text(r.stressLookAlike!),
                ),
              ),

            const SizedBox(height: 8),

            // ── Advisory (now wired to AdvisoryService) ──
            Card(
              color: colorScheme.secondaryContainer,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.medical_services_rounded,
                          color: colorScheme.onSecondaryContainer,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Treatment Advisory',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                            color: colorScheme.onSecondaryContainer,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    if (_advisoryLoading)
                      const Row(
                        children: [
                          SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                          SizedBox(width: 8),
                          Text('Loading advisory...'),
                        ],
                      )
                    else
                      Text(
                        _advisoryText ?? r.advisory,
                        style: TextStyle(
                          color: colorScheme.onSecondaryContainer,
                          height: 1.5,
                        ),
                      ),
                  ],
                ),
              ),
            ),

            // ── Save status indicator ──
            if (_saved)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.check_circle_outline, size: 16, color: colorScheme.primary),
                    const SizedBox(width: 4),
                    Text(
                      'Saved to history',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: colorScheme.primary,
                      ),
                    ),
                  ],
                ),
              ),
            if (_saveError != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.error_outline, size: 16, color: Colors.red),
                    const SizedBox(width: 4),
                    Flexible(
                      child: Text(
                        'Save failed: $_saveError',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: Colors.red,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

            const SizedBox(height: 16),

            // ── Action buttons ──
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _toggleVoice,
                    icon: Icon(_isSpeaking
                        ? Icons.stop_rounded
                        : Icons.volume_up_rounded),
                    label: Text(_isSpeaking ? 'Stop' : 'Listen'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: () {
                      Navigator.popUntil(context, (route) => route.isFirst);
                    },
                    icon: const Icon(Icons.camera_alt_rounded),
                    label: const Text('Scan Again'),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 24),

            // ── Latency Chips ──
            Center(
              child: Wrap(
                spacing: 8,
                alignment: WrapAlignment.center,
                children: [
                  Chip(
                    label: Text('Inference: ${r.inferenceLatencyMs}ms'),
                    avatar: const Icon(Icons.speed_rounded, size: 16),
                  ),
                  if (_gradCamLatencyMs != null)
                    Chip(
                      label: Text('Grad-CAM: ${_gradCamLatencyMs}ms'),
                      avatar: const Icon(Icons.analytics_rounded, size: 16),
                    ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // ── Raw Debug Panel ──
            ExpansionTile(
              title: const Text('Raw Debug Scores', style: TextStyle(fontWeight: FontWeight.bold)),
              children: [
                _buildDebugList('Disease', r.rawDiseaseScores),
                _buildDebugList('Severity', r.rawSeverityScores),
                _buildDebugList('Stress Flag', r.rawStressScores),
              ],
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildDebugList(String title, Map<String, double> scores) {
    final sortedScores = scores.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 4),
          ...sortedScores.map((e) => Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(e.key, style: const TextStyle(fontSize: 12)),
              Text(e.value.toStringAsFixed(4), style: const TextStyle(fontSize: 12, fontFamily: 'monospace')),
            ],
          )),
        ],
      ),
    );
  }
}
