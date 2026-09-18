import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../../main.dart' show cameras;
import '../inference/inference_service.dart';
import 'result_screen.dart';

/// Full-screen camera capture wired to the device camera.
/// Takes a photo and passes it to the inference pipeline.
class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isCapturing = false;
  bool _isCameraReady = false;
  FlashMode _flashMode = FlashMode.auto;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;

    if (state == AppLifecycleState.inactive) {
      ctrl.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    if (cameras.isEmpty) {
      setState(() => _errorMessage = 'No cameras available on this device.');
      return;
    }

    // Use the back camera (first in the list is usually rear)
    final backCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      backCamera,
      // Medium resolution balances quality vs. speed for inference
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    _controller = controller;

    try {
      await controller.initialize();
      await controller.setFlashMode(_flashMode);

      if (!mounted) return;
      setState(() {
        _isCameraReady = true;
        _errorMessage = null;
      });
    } on CameraException catch (e) {
      setState(() {
        _errorMessage = 'Camera error: ${e.description}';
      });
    }
  }

  Future<void> _captureAndAnalyze() async {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized || _isCapturing) return;

    setState(() => _isCapturing = true);

    try {
      // Capture the image
      final XFile imageFile = await ctrl.takePicture();
      debugPrint('Photo captured: ${imageFile.path}');

      if (!mounted) return;

      // Run inference (stub for now — returns dummy data)
      final result = await InferenceService.runDiagnosis(imageFile.path);

      if (!mounted) return;

      // Navigate to results screen
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            imagePath: imageFile.path,
            diagnosisResult: result,
          ),
        ),
      );
    } catch (e) {
      debugPrint('Capture error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Capture failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  void _toggleFlash() {
    final modes = [FlashMode.auto, FlashMode.always, FlashMode.off];
    final nextIndex = (modes.indexOf(_flashMode) + 1) % modes.length;
    _flashMode = modes[nextIndex];
    _controller?.setFlashMode(_flashMode);
    setState(() {});
  }

  IconData get _flashIcon {
    switch (_flashMode) {
      case FlashMode.auto:
        return Icons.flash_auto_rounded;
      case FlashMode.always:
        return Icons.flash_on_rounded;
      case FlashMode.off:
        return Icons.flash_off_rounded;
      default:
        return Icons.flash_auto_rounded;
    }
  }

  String get _flashLabel {
    switch (_flashMode) {
      case FlashMode.auto:
        return 'Auto';
      case FlashMode.always:
        return 'On';
      case FlashMode.off:
        return 'Off';
      default:
        return 'Auto';
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Camera preview ──
          if (_isCameraReady && _controller != null)
            Center(
              child: AspectRatio(
                aspectRatio: 1 / _controller!.value.aspectRatio,
                child: CameraPreview(_controller!),
              ),
            )
          else if (_errorMessage != null)
            Center(
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.error_outline, color: Colors.red, size: 64),
                    const SizedBox(height: 16),
                    Text(
                      _errorMessage!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  ],
                ),
              ),
            )
          else
            const Center(
              child: CircularProgressIndicator(color: Colors.white),
            ),

          // ── Top controls ──
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                child: Row(
                  children: [
                    // Back button
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_rounded),
                      color: Colors.white,
                      style: IconButton.styleFrom(
                        backgroundColor: Colors.black38,
                      ),
                    ),
                    const Spacer(),
                    // Flash toggle
                    _ControlPill(
                      icon: _flashIcon,
                      label: _flashLabel,
                      onTap: _toggleFlash,
                    ),
                  ],
                ),
              ),
            ),
          ),

          // ── Viewfinder guide overlay ──
          if (_isCameraReady)
            Center(
              child: Container(
                width: MediaQuery.of(context).size.width * 0.78,
                height: MediaQuery.of(context).size.width * 0.78,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.4),
                    width: 2,
                  ),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Text(
                        'Center the leaf in the frame',
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // ── Loading Overlay ──
          if (_isCapturing)
            Positioned.fill(
              child: Container(
                color: Colors.black54,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: const [
                    CircularProgressIndicator(color: Colors.white),
                    SizedBox(height: 16),
                    Text(
                      'Running inference...',
                      style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),
            ),

          // ── Bottom capture bar ──
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: SafeArea(
              child: Container(
                padding: const EdgeInsets.symmetric(vertical: 24),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      Colors.black.withValues(alpha: 0.8),
                      Colors.transparent,
                    ],
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    // Gallery (placeholder)
                    IconButton(
                      onPressed: () {
                        // TODO: Pick from gallery
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Gallery picker — coming soon'),
                          ),
                        );
                      },
                      icon: const Icon(Icons.photo_library_rounded),
                      color: Colors.white70,
                      iconSize: 28,
                    ),

                    // Shutter button
                    GestureDetector(
                      onTap: _isCameraReady ? _captureAndAnalyze : null,
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 150),
                        width: 76,
                        height: 76,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 4),
                          color: _isCapturing
                              ? colorScheme.primary.withValues(alpha: 0.6)
                              : Colors.transparent,
                        ),
                        child: Center(
                          child: _isCapturing
                              ? const SizedBox(
                                  width: 28,
                                  height: 28,
                                  child: CircularProgressIndicator(
                                    color: Colors.white,
                                    strokeWidth: 3,
                                  ),
                                )
                              : Container(
                                  width: 60,
                                  height: 60,
                                  decoration: const BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: Colors.white,
                                  ),
                                ),
                        ),
                      ),
                    ),

                    // Placeholder for symmetry
                    const SizedBox(width: 48),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// A small pill-shaped control widget for the top bar.
class _ControlPill extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ControlPill({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black38,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white, size: 18),
            const SizedBox(width: 4),
            Text(
              label,
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}
