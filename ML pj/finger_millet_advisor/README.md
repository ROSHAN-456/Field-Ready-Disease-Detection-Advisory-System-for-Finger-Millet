# Millet Doctor — Finger Millet Disease Detection & Advisory App

An offline, AI-powered mobile application for detecting diseases in finger millet
and related crops (millets, groundnut, banana) using leaf photos.

## Features
- 📷 Camera-based leaf disease detection
- 🤖 On-device TFLite inference (<10MB model, sub-second)
- 🔬 Grad-CAM explainability heatmaps
- 💊 ICAR/KVK-sourced treatment advisories
- 🔊 Voice advisory in regional languages
- 💾 Local SQLite diagnosis history
- 📶 100% offline — no cloud, no server

## Project Structure
```
lib/
├── main.dart                              # App entry point
└── features/
    ├── home/
    │   └── home_screen.dart               # Home screen with Scan Leaf CTA
    ├── capture/
    │   ├── capture_screen.dart            # Camera capture (fully wired)
    │   └── result_screen.dart             # Diagnosis result display
    ├── inference/
    │   └── inference_service.dart          # TFLite inference (stub)
    ├── explainability/
    │   └── gradcam_service.dart            # Grad-CAM heatmap (stub)
    ├── advisory/
    │   └── advisory_service.dart           # Treatment lookup (stub)
    ├── voice/
    │   └── voice_service.dart             # TTS voice output
    ├── storage/
    │   └── storage_service.dart           # SQLite CRUD
    └── history_ui/
        └── history_screen.dart            # Diagnosis history list
```

## Setup
1. Install [Flutter SDK](https://docs.flutter.dev/get-started/install/windows)
2. Run `flutter pub get`
3. Connect a device or start an emulator
4. Run `flutter run`

## Tech Stack
- Flutter (Dart)
- TensorFlow Lite (tflite_flutter)
- Camera (camera plugin)
- SQLite (sqflite)
- Text-to-Speech (flutter_tts)
- Min SDK: 24 (Android 7.0+)
