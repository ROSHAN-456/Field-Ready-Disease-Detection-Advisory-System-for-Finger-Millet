import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';

import 'features/capture/capture_screen.dart';
import 'features/history_ui/history_screen.dart';
import 'features/home/home_screen.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock to portrait for consistent camera + UI experience
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Initialize available cameras
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    debugPrint('Camera initialization error: ${e.code} - ${e.description}');
    cameras = [];
  }

  runApp(const MilletDoctorApp());
}

class MilletDoctorApp extends StatelessWidget {
  const MilletDoctorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Millet Doctor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF2E7D32), // Deep green — crop theme
        brightness: Brightness.light,
        fontFamily: 'Roboto',
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          elevation: 0,
        ),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF66BB6A),
        brightness: Brightness.dark,
        fontFamily: 'Roboto',
      ),
      themeMode: ThemeMode.system,
      initialRoute: '/',
      routes: {
        '/': (context) => const HomeScreen(),
        '/capture': (context) => const CaptureScreen(),
        '/history': (context) => const HistoryScreen(),
      },
    );
  }
}
