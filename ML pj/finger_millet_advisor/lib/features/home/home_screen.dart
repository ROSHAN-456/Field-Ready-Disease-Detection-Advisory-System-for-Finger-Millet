import 'package:flutter/material.dart';

/// Home screen with a prominent "Scan Leaf" button.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final size = MediaQuery.of(context).size;

    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              colorScheme.primaryContainer.withValues(alpha: 0.3),
              colorScheme.surface,
              colorScheme.surface,
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              const SizedBox(height: 48),

              // ── App icon ──
              Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  color: colorScheme.primaryContainer,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: colorScheme.primary.withValues(alpha: 0.3),
                      blurRadius: 20,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: Icon(
                  Icons.eco_rounded,
                  size: 52,
                  color: colorScheme.primary,
                ),
              ),

              const SizedBox(height: 24),

              // ── Title ──
              Text(
                'Millet Doctor',
                style: theme.textTheme.headlineLarge?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 8),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 48),
                child: Text(
                  'AI-powered crop disease detection\nfor finger millet & related crops',
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    height: 1.5,
                  ),
                ),
              ),

              const Spacer(),

              // ── Feature pills ──
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  alignment: WrapAlignment.center,
                  children: [
                    _FeatureChip(
                      icon: Icons.offline_bolt_rounded,
                      label: '100% Offline',
                      colorScheme: colorScheme,
                    ),
                    _FeatureChip(
                      icon: Icons.speed_rounded,
                      label: 'Sub-second',
                      colorScheme: colorScheme,
                    ),
                    _FeatureChip(
                      icon: Icons.record_voice_over_rounded,
                      label: 'Voice Advisory',
                      colorScheme: colorScheme,
                    ),
                    _FeatureChip(
                      icon: Icons.visibility_rounded,
                      label: 'Grad-CAM',
                      colorScheme: colorScheme,
                    ),
                  ],
                ),
              ),

              const Spacer(),

              // ── Scan Leaf CTA ──
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: SizedBox(
                  width: double.infinity,
                  height: 60,
                  child: FilledButton.icon(
                    onPressed: () => Navigator.pushNamed(context, '/capture'),
                    icon: const Icon(Icons.camera_alt_rounded, size: 24),
                    label: const Text(
                      'Scan Leaf',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    style: FilledButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // ── History shortcut ──
              TextButton.icon(
                onPressed: () {
                  Navigator.pushNamed(context, '/history');
                },
                icon: Icon(
                  Icons.history_rounded,
                  color: colorScheme.primary,
                ),
                label: Text(
                  'View Past Diagnoses',
                  style: TextStyle(color: colorScheme.primary),
                ),
              ),

              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}

class _FeatureChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final ColorScheme colorScheme;

  const _FeatureChip({
    required this.icon,
    required this.label,
    required this.colorScheme,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: colorScheme.outlineVariant.withValues(alpha: 0.5),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: colorScheme.primary),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: colorScheme.onSurface,
            ),
          ),
        ],
      ),
    );
  }
}
