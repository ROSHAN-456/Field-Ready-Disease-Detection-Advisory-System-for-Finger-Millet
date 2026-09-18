/// Voice module — on-device Text-to-Speech.
///
/// Uses Android's native TTS engine via flutter_tts plugin
/// to read diagnosis and advisory aloud in regional languages.
library;

import 'package:flutter_tts/flutter_tts.dart';

class VoiceService {
  static final FlutterTts _tts = FlutterTts();
  static bool _initialized = false;

  /// Initialize TTS with the given language code.
  /// Default: 'en-IN' (English, India).
  /// Supported: 'kn-IN' (Kannada), 'hi-IN' (Hindi), 'ta-IN' (Tamil), etc.
  static Future<void> init({String languageCode = 'en-IN'}) async {
    await _tts.setLanguage(languageCode);
    await _tts.setSpeechRate(0.45); // Slightly slower for clarity
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
    _initialized = true;
  }

  /// Speak the given text aloud.
  static Future<void> speak(String text) async {
    if (!_initialized) await init();
    await _tts.speak(text);
  }

  /// Stop any ongoing speech.
  static Future<void> stop() async {
    await _tts.stop();
  }

  /// Check if a specific language is available on this device.
  static Future<bool> isLanguageAvailable(String langCode) async {
    final result = await _tts.isLanguageAvailable(langCode);
    return result == 1 || result == true;
  }

  /// Get list of available languages on this device.
  static Future<List<String>> getAvailableLanguages() async {
    final langs = await _tts.getLanguages;
    return List<String>.from(langs ?? []);
  }
}
