/// Storage module — SQLite database for diagnosis history.
///
/// Stores each scan result with timestamp, image path,
/// disease type, severity, confidence, and advisory text.
library;

import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class StorageService {
  static Database? _db;
  static const String _tableName = 'diagnosis_history';

  /// Open (or create) the local SQLite database.
  static Future<Database> get database async {
    if (_db != null) return _db!;
    final dbPath = await getDatabasesPath();
    _db = await openDatabase(
      join(dbPath, 'millet_doctor.db'),
      version: 2,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE $_tableName (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            severity TEXT NOT NULL,
            confidence REAL NOT NULL,
            stress_flag TEXT NOT NULL DEFAULT 'unknown',
            stress_lookalike TEXT,
            advisory TEXT NOT NULL,
            crop_type TEXT DEFAULT 'finger_millet',
            latitude REAL,
            longitude REAL,
            synced INTEGER DEFAULT 0
          )
        ''');
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute('ALTER TABLE $_tableName ADD COLUMN stress_flag TEXT NOT NULL DEFAULT \'unknown\'');
        }
      },
    );
    return _db!;
  }

  /// Save a diagnosis record to the local database.
  static Future<int> saveDiagnosis({
    required String imagePath,
    required String diseaseName,
    required String severity,
    required double confidence,
    required String stressFlag,
    String? stressLookAlike,
    required String advisory,
    String cropType = 'finger_millet',
  }) async {
    final db = await database;
    return db.insert(_tableName, {
      'timestamp': DateTime.now().toIso8601String(),
      'image_path': imagePath,
      'disease_name': diseaseName,
      'severity': severity,
      'confidence': confidence,
      'stress_flag': stressFlag,
      'stress_lookalike': stressLookAlike,
      'advisory': advisory,
      'crop_type': cropType,
      'synced': 0,
    });
  }

  /// Retrieve all diagnosis records, most recent first.
  static Future<List<Map<String, dynamic>>> getAllDiagnoses() async {
    final db = await database;
    return db.query(
      _tableName,
      orderBy: 'timestamp DESC',
    );
  }

  /// Get diagnosis records that haven't been synced yet.
  static Future<List<Map<String, dynamic>>> getUnsyncedDiagnoses() async {
    final db = await database;
    return db.query(
      _tableName,
      where: 'synced = ?',
      whereArgs: [0],
      orderBy: 'timestamp DESC',
    );
  }

  /// Mark a record as synced.
  static Future<int> markSynced(int id) async {
    final db = await database;
    return db.update(
      _tableName,
      {'synced': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  /// Delete a diagnosis record by ID.
  static Future<int> deleteDiagnosis(int id) async {
    final db = await database;
    return db.delete(_tableName, where: 'id = ?', whereArgs: [id]);
  }

  /// Get total number of diagnoses.
  static Future<int> getDiagnosisCount() async {
    final db = await database;
    final result = await db.rawQuery('SELECT COUNT(*) as count FROM $_tableName');
    return result.first['count'] as int;
  }
}
