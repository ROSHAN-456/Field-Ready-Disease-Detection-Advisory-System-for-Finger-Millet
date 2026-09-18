/// History Detail UI — displays a full saved diagnosis record.
///
/// Receives a raw database row and renders
/// a read-only view with image, disease, severity, advisory, and timestamp.
library;

import 'dart:io';
import 'package:flutter/material.dart';

import '../voice/voice_service.dart';

class HistoryDetailScreen extends StatefulWidget {
  final Map<String, dynamic> record;

  const HistoryDetailScreen({super.key, required this.record});

  @override
  State<HistoryDetailScreen> createState() => _HistoryDetailScreenState();
}

class _HistoryDetailScreenState extends State<HistoryDetailScreen> {
  bool _isSpeaking = false;

  Map<String, dynamic> get r => widget.record;

  @override
  void dispose() {
    if (_isSpeaking) VoiceService.stop();
    super.dispose();
  }

  Future<void> _toggleVoice() async {
    if (_isSpeaking) {
      await VoiceService.stop();
      if (mounted) setState(() => _isSpeaking = false);
    } else {
      final text = 'Disease: ${r['disease_name']}. '
          'Severity: ${r['severity']}. '
          '${r['advisory'] ?? ''}';
      setState(() => _isSpeaking = true);
      await VoiceService.speak(text);
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

  String _formatDate(String? ts) {
    if (ts == null) return '';
    try {
      final dt = DateTime.parse(ts);
      return '${dt.day}/${dt.month}/${dt.year} ${dt.hour}:${dt.minute.toString().padLeft(2, '0')}';
    } catch (_) {
      return ts;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final severity = (r['severity'] ?? '').toString();
    final sevColor = _severityColor(severity);
    final imagePath = r['image_path'] as String?;
    final imageFile = imagePath != null ? File(imagePath) : null;
    final imageExists = imageFile != null && imageFile.existsSync();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Diagnosis Detail'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Image ──
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: AspectRatio(
                aspectRatio: 1,
                child: imageExists
                    ? Image.file(imageFile, fit: BoxFit.cover)
                    : Container(
                        color: colorScheme.surfaceContainerHighest,
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.image_not_supported_rounded,
                                size: 48, color: colorScheme.outlineVariant),
                            const SizedBox(height: 8),
                            Text(
                              'Image no longer available',
                              style: TextStyle(color: colorScheme.outlineVariant),
                            ),
                          ],
                        ),
                      ),
              ),
            ),
            const SizedBox(height: 16),

            // ── Disease ──
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
                            'Disease: ${r['disease_name'] ?? 'Unknown'}',
                            style: theme.textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ],
                    ),
                    if (r['confidence'] != null) ...[
                      const SizedBox(height: 8),
                      LinearProgressIndicator(
                        value: (r['confidence'] as num).toDouble(),
                        backgroundColor: colorScheme.surfaceContainerHighest,
                        color: colorScheme.primary,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Confidence: ${((r['confidence'] as num) * 100).toStringAsFixed(1)}%',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),

            // ── Severity ──
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Icon(_severityIcon(severity), color: sevColor, size: 28),
                    const SizedBox(width: 8),
                    Text(
                      'Severity: ${severity.toUpperCase()}',
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        color: sevColor,
                        fontSize: 18,
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
                child: Row(
                  children: [
                    Icon(Icons.grass_rounded, color: colorScheme.secondary, size: 28),
                    const SizedBox(width: 8),
                    Text(
                      'Classification: ${(r['stress_flag'] ?? 'unknown').toString().toUpperCase()}',
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        color: colorScheme.secondary,
                        fontSize: 18,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // ── Stress look-alike ──
            if (r['stress_lookalike'] != null && (r['stress_lookalike'] as String).isNotEmpty)
              Card(
                child: ListTile(
                  leading: Icon(Icons.info_outline_rounded, color: colorScheme.tertiary),
                  title: const Text('Stress Look-alike'),
                  subtitle: Text(r['stress_lookalike']),
                ),
              ),

            const SizedBox(height: 8),

            // ── Advisory ──
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
                    Text(
                      r['advisory'] ?? 'No advisory available.',
                      style: TextStyle(
                        color: colorScheme.onSecondaryContainer,
                        height: 1.5,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 8),

            // ── Timestamp ──
            Center(
              child: Chip(
                label: Text(_formatDate(r['timestamp'])),
                avatar: const Icon(Icons.access_time_rounded, size: 16),
              ),
            ),

            const SizedBox(height: 16),

            // ── Play advisory button ──
            OutlinedButton.icon(
              onPressed: _toggleVoice,
              icon: Icon(_isSpeaking ? Icons.stop_rounded : Icons.volume_up_rounded),
              label: Text(_isSpeaking ? 'Stop' : 'Play Advisory'),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),

            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
