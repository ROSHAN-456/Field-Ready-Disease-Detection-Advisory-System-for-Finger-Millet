/// Advisory module — treatment recommendation lookup.
///
/// Queries local ICAR/KVK-sourced database for treatment
/// recommendations matched to disease + severity.
library;

import 'dart:convert';
import 'package:flutter/services.dart';

class AdvisoryService {
  static Map<String, dynamic>? _db;

  /// Loads the database if not already loaded.
  static Future<void> _loadDb() async {
    if (_db != null) return;
    try {
      final jsonString = await rootBundle.loadString('assets/advisory_db/advisory.json');
      _db = json.decode(jsonString);
    } catch (e) {
      throw Exception('Failed to load advisory database: $e');
    }
  }

  /// Looks up treatment recommendation for a given disease and severity.
  static Future<String> getRecommendation({
    required String diseaseName,
    required String severity,
    String cropType = 'finger_millet',
  }) async {
    await _loadDb();

    // Normalize keys
    final dKey = diseaseName.toLowerCase().replaceAll(' ', '_');
    final sKey = severity.toLowerCase().replaceAll(' ', '_');

    // Handle stress conditions (they are passed as diseaseName starting with 'stress_')
    // and healthy
    final isStress = dKey.startsWith('stress_');
    final isHealthy = dKey == 'healthy_leaf' || dKey == 'healthy';

    String lookupDisease = dKey;
    String lookupSeverity = sKey;

    if (isHealthy) {
      lookupDisease = 'healthy';
      lookupSeverity = 'default';
    } else if (isStress) {
      // Stress labels are already bare keys in the DB (stress_sunburn, etc.)
      lookupSeverity = 'default';
    } else {
      // The disease head predicts a combined "<disease>_<severity>" label
      // (e.g. "blast_mild", "leaf_blight_severe"), but the advisory DB is
      // keyed by bare disease name with severity as a nested key. Strip the
      // trailing severity suffix so the lookup actually matches a DB entry.
      const severitySuffixes = ['_mild', '_moderate', '_severe'];
      for (final suffix in severitySuffixes) {
        if (dKey.endsWith(suffix)) {
          lookupDisease = dKey.substring(0, dKey.length - suffix.length);
          break;
        }
      }
      // Prefer the severity embedded in the disease label (authoritative,
      // since it comes from the same head as the disease name) but fall
      // back to the separately-predicted severity head if none was found.
      for (final suffix in severitySuffixes) {
        if (dKey.endsWith(suffix)) {
          lookupSeverity = suffix.substring(1); // drop leading underscore
          break;
        }
      }
    }

    final diseaseEntry = _db![lookupDisease];
    
    // Fallback if disease not found
    if (diseaseEntry == null) {
       final fallback = _db!['fallback']['default'];
       return '${fallback['recommendation']}\n\nDosage: ${fallback['dosage']}\nSource: ${fallback['source']}';
    }

    // Lookup severity, fallback to default if specific severity not found
    final severityEntry = diseaseEntry[lookupSeverity] ?? diseaseEntry['default'] ?? _db!['fallback']['default'];

    final rec = severityEntry['recommendation'];
    final dosage = severityEntry['dosage'];
    final source = severityEntry['source'];

    return '$rec\n\nDosage: $dosage\nSource: $source';
  }
}
