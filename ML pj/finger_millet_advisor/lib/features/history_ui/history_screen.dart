/// History UI — displays past diagnosis records.
///
/// Reads from the local SQLite database and displays
/// a scrollable list of past scans with disease, severity, and date.
library;

import 'package:flutter/material.dart';

import '../storage/storage_service.dart';
import 'history_detail_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<Map<String, dynamic>> _records = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final records = await StorageService.getAllDiagnoses();
    setState(() {
      _records = records;
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Diagnosis History')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _records.isEmpty
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.history_rounded,
                        size: 64,
                        color: theme.colorScheme.outlineVariant,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'No diagnoses yet',
                        style: theme.textTheme.titleMedium?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Scan a leaf to get started',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.outline,
                        ),
                      ),
                    ],
                  ),
                )
              : RefreshIndicator(
                  onRefresh: _loadHistory,
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: _records.length,
                    itemBuilder: (context, index) {
                      final r = _records[index];
                      return Card(
                        margin: const EdgeInsets.only(bottom: 8),
                        child: ListTile(
                          leading: CircleAvatar(
                            backgroundColor: _sevColor(r['severity'] ?? ''),
                            child: Icon(
                              _sevIcon(r['severity'] ?? ''),
                              color: Colors.white,
                              size: 20,
                            ),
                          ),
                          title: Text(r['disease_name'] ?? 'Unknown'),
                          subtitle: Text(
                            '${(r['severity'] ?? '').toString().toUpperCase()} · '
                            '${(r['stress_flag'] ?? 'UNKNOWN').toString().toUpperCase()}\n'
                            '${_formatDate(r['timestamp'])}',
                          ),
                          trailing: const Icon(Icons.chevron_right_rounded),
                          onTap: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (_) => HistoryDetailScreen(record: r),
                              ),
                            );
                          },
                        ),
                      );
                    },
                  ),
                ),
    );
  }

  Color _sevColor(String severity) {
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

  IconData _sevIcon(String severity) {
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
}
