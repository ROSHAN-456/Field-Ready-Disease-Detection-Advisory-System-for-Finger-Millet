@echo off
REM ============================================================
REM  Finger Millet Pipeline — One-Command Rerun
REM  Usage:
REM    run_full_pipeline.bat                    <- uses synthetic data
REM    run_full_pipeline.bat "d:\real_dataset"  <- uses your real images
REM    run_full_pipeline.bat "d:\real_dataset" 800  <- real data + 800 imgs/class target
REM ============================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set RAW_DATA=%~1
set AUG_TARGET=%~2

if "%AUG_TARGET%"=="" set AUG_TARGET=500

echo.
echo ============================================================
echo   FINGER MILLET DISEASE DETECTION — FULL PIPELINE
echo ============================================================
echo   Raw data  : %RAW_DATA%
echo   Aug target: %AUG_TARGET% images/class
echo   Started   : %DATE% %TIME%
echo ============================================================
echo.

if "%RAW_DATA%"=="" (
    echo [INFO] No --raw_data supplied. Running on existing synthetic dataset.
    python run_pipeline.py --aug_target %AUG_TARGET%
) else (
    python run_pipeline.py --raw_data "%RAW_DATA%" --aug_target %AUG_TARGET%
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAILED] Pipeline exited with error code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   [DONE] Pipeline finished successfully.
echo   TFLite model : %SCRIPT_DIR%exported_model\finger_millet_model.tflite
echo   Eval reports : %SCRIPT_DIR%eval_results\
echo ============================================================
