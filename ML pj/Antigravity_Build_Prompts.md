# Antigravity Build Prompts — Finger Millet Disease Detection & Advisory App

Paste these prompts into Antigravity **one at a time, in order**. Let each phase finish
(and verify the output) before moving to the next. Each prompt is self-contained but
references the same project context, so Antigravity keeps consistent architecture across phases.

---

## PHASE 0 — Project Context (paste this first, every new session)

```
I am building an offline, mobile disease-detection and advisory app for finger millet
(a staple crop), plus related crops like millets, groundnut, and banana.

CORE REQUIREMENTS:
- Runs fully offline on a mid-range Android phone. No cloud, no server, no IoT sensors.
- Input: a single leaf photo taken with the phone camera in poor/cluttered farm lighting
  (not clean lab images).
- Output in one pass: (1) disease type, (2) severity stage — mild/moderate/severe,
  (3) non-disease stress look-alikes (nutrient deficiency, pest damage, sunburn).
- Explainability: Grad-CAM heatmap overlay showing which leaf regions drove the diagnosis.
- Advisory: treatment recommendation pulled from a local ICAR/KVK-sourced database,
  matched to diagnosis + severity.
- Voice: recommendation read aloud on-device in the farmer's regional language.
- Storage: diagnosis history logged locally in SQLite, with optional sync later.

TECH STACK (do not deviate without telling me why):
- Model: MobileNetV3 or EfficientNet-Lite CNN backbone, fine-tuned for
  disease + severity + stress classification.
- On-device runtime: TensorFlow Lite, quantized, target <10MB model size,
  sub-second inference.
- Explainability: Grad-CAM computed and rendered on-device.
- App: Android, built in Flutter (so we get one codebase and easy TFLite + TTS plugins).
- Advisory content: local on-device database (SQLite or bundled JSON), sourced from
  ICAR/KVK treatment guidelines.
- Voice: on-device TTS engine (Android's native TTS via Flutter plugin), regional
  language support.
- Storage: SQLite for diagnosis history.

Acknowledge this context and wait for my next instruction — do not start building yet.
```

---

## PHASE 1 — Project Scaffolding

```
Scaffold the Flutter project for this app.

1. Create a new Flutter project named `finger_millet_advisor` with a clean
   feature-based folder structure: /capture, /inference, /explainability,
   /advisory, /voice, /storage, /history_ui.
2. Add dependencies: camera, tflite_flutter, sqflite, flutter_tts, path_provider.
3. Set minSdkVersion appropriately for TFLite + camera support on mid-range phones.
4. Create a placeholder home screen with a single "Scan Leaf" button that opens
   the camera capture screen (build the capture screen fully, wired to the device
   camera, but leave inference as a stub returning dummy data for now).
5. Confirm the app builds and runs on an emulator before moving on.
```

---

## PHASE 2 — Dataset Collection & Organization

```
Help me set up the dataset pipeline for training, before any model work.

1. Design a folder/label schema covering: disease classes (name them for finger
   millet — blast, blight, etc.), 3 severity levels (mild/moderate/severe) per
   disease, and 3 non-disease stress classes (nutrient deficiency, pest damage,
   sunburn).
2. Write a Python script to validate a dataset folder against this schema
   (checks class balance, image counts, corrupt files, duplicate detection).
3. Write an augmentation script simulating real field conditions: cluttered
   backgrounds, uneven lighting, blur, phone-camera artifacts — since training
   data must generalize beyond clean lab images.
4. Write a script to split into train/val/test with stratification by class.
5. Output a short data-quality report (class counts, image resolution stats)
   as a markdown file.
```

---

## PHASE 3 — Model Architecture & Training

```
Build the training pipeline in Python/TensorFlow.

1. Implement a multi-head model: MobileNetV3-Small (or EfficientNet-Lite0)
   backbone, with three output heads — disease type, severity stage, and
   stress-vs-disease flag. Use transfer learning from ImageNet weights.
2. Write the training script with: class-weighted loss (for imbalance),
   early stopping, checkpointing, and a config file for hyperparameters.
3. Add an evaluation script producing per-class precision/recall/F1 and a
   confusion matrix for each head.
4. Add a "domain-shift benchmark" script: evaluate the trained model separately
   on (a) lab-style/clean images and (b) field-condition images, and output a
   comparison report — this is one of our required deliverables.
5. Target: keep the model small enough that post-quantization it stays under 10MB.
```

---

## PHASE 4 — TFLite Conversion & Quantization

```
Convert the trained model for on-device deployment.

1. Write a conversion script using TensorFlow Lite converter with full integer
   post-training quantization (with a representative dataset for calibration).
2. Verify final model size is under 10MB; if not, suggest architecture/pruning
   changes and iterate.
3. Benchmark inference latency on a simulated mid-range CPU profile — confirm
   sub-second inference.
4. Export the .tflite file plus a label map (JSON) for disease, severity, and
   stress classes.
5. Write a small Python sanity-check script that loads the .tflite model and
   runs inference on a handful of test images, comparing outputs to the
   original Keras model to confirm no major accuracy loss from quantization.
```

---

## PHASE 5 — On-Device Inference Integration (Flutter)

```
Wire the real model into the Flutter app, replacing the Phase 1 stub.


1. Integrate tflite_flutter to load the .tflite model and label map from assets.
2. Implement the inference pipeline: preprocess captured image (resize, normalize)
   → run inference → parse the three output heads → return a structured
   DiagnosisResult object (disease, severity, stressFlag, confidence scores).
3. Add error handling for low-confidence predictions (e.g., prompt farmer to
   retake photo if the leaf isn't clearly visible).
4. Show the raw diagnosis result on screen (text only, no styling yet) so we can
   verify correctness end-to-end from camera → model → result before building UI polish.
```

---

## PHASE 6 — Grad-CAM Explainability Overlay

```
Add the explainability layer.

1. Implement Grad-CAM computation compatible with the quantized TFLite model
   (or, if TFLite doesn't expose intermediate activations cleanly, use a
   parallel float model loaded just for Grad-CAM, and explain the tradeoff to me).
2. Render the Grad-CAM heatmap as a semi-transparent overlay on the original
   leaf photo.
3. Add this overlay to the diagnosis result screen, so the farmer sees the
   photo with highlighted regions next to the diagnosis text.
4. Confirm heatmap generation still keeps total time-to-result under ~2 seconds
   on a mid-range device profile.
```

---

## PHASE 7 — Advisory Database (ICAR/KVK)

```
Build the local advisory content system.

1. Design a SQLite schema (or bundled JSON, your recommendation) mapping
   {disease, severity} → treatment recommendation text, sourced from ICAR/KVK
   guidance. Include fields for: recommendation text, dosage/timing if applicable,
   and a source citation field.
2. Populate it with placeholder entries for each disease/severity combination
   we defined in Phase 2 — I will supply/verify the real ICAR/KVK text afterward.
3. Write the lookup logic: given a DiagnosisResult, fetch the matching advisory
   entry (with a sensible fallback if severity is borderline or stress-flagged
   rather than disease-flagged).
4. Display the advisory text on the result screen beneath the Grad-CAM overlay.
```

---

## PHASE 8 — Voice Output (Regional Language)

```
Add regional-language voice output for the advisory.

1. Integrate flutter_tts, configure it for [name your target regional
   language(s), e.g., Kannada/Tamil/Telugu/Hindi].
2. Add a "Play Advisory" button on the result screen that reads the advisory
   text aloud using on-device TTS.
3. Handle the case where the device doesn't have the target language TTS voice
   installed — show a graceful fallback message and a link to Android's TTS
   language settings.
4. Test with sample advisory text to confirm pronunciation is acceptable.
```

---

## PHASE 9 — Local Storage & History

```
Implement diagnosis history logging.

1. Design a SQLite table for diagnosis history: timestamp, image path
   (or thumbnail), disease, severity, stress flag, advisory shown.
2. Save every completed diagnosis automatically to this table.
3. Build a simple history screen: a scrollable list of past diagnoses, each
   tappable to reopen the full result (photo + heatmap + advisory).
4. Add a placeholder "sync" function (no-op for now, just a stub) noting where
   cloud sync would hook in later if connectivity becomes available.
```

---

## PHASE 10 — End-to-End Testing & Packaging

```
Final integration pass before packaging.

1. Run through the full flow end-to-end: capture → inference → Grad-CAM →
   advisory → voice → history save, on an emulator and (if available) a real
   mid-range Android device.
2. Check total model + app size stays within a reasonable APK size given the
   <10MB model constraint.
3. Fix any UI rough edges on the capture and result screens (basic, clean,
   readable for a non-technical farmer — large buttons, minimal text, icons
   over jargon).
4. Build a release APK and confirm it installs and runs fully offline
   (disable network on the test device to verify no functionality breaks).
5. Summarize what's done vs. what still needs real data (trained weights,
   real ICAR/KVK content, real regional-language TTS testing) so I know exactly
   what's placeholder vs. production-ready.
```

---

### Notes on using this
- Antigravity works best when you let it finish and **verify** each phase (run the
  emulator, check the output) before pasting the next prompt — don't paste all of
  them in one message.
- Phases 2–4 (dataset, training, quantization) are Python/ML work; Phases 1, 5–10 are
  the Flutter app. If Antigravity tries to mix these into one environment, tell it to
  keep the ML pipeline in a separate `/ml` folder from the Flutter `/app` folder.
- Swap in your real disease names, regional language(s), and actual ICAR/KVK advisory
  text in Phases 2 and 7 — the prompts use placeholders on purpose since only you
  have the domain specifics.
  