@echo off
REM Manual Training Script for SKU Predictor
REM Allows manual execution of training processes

echo ========================================
echo SKU Predictor Manual Training
echo ========================================

:menu
echo.
echo Please select training mode:
echo.
echo 1. Weekly Incremental Training (20-60 minutes)
echo 2. Monthly Full Retraining (5-14 hours)
echo 3. Test Incremental Training (small dataset)
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto weekly
if "%choice%"=="2" goto monthly
if "%choice%"=="3" goto test
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
goto menu

:weekly
echo.
echo ========================================
echo Starting Weekly Incremental Training
echo ========================================
echo ⏱️ Expected duration: 20-60 minutes
echo 📊 Processing: Last 7 days of data
echo.
echo Starting training...
python scripts\weekly_incremental_training.py
if %errorlevel% equ 0 (
    echo ✅ Weekly training completed successfully!
) else (
    echo ❌ Weekly training failed. Check logs for details.
)
goto end

:monthly
echo.
echo ========================================
echo Starting Monthly Full Retraining
echo ========================================
echo ⏱️ Expected duration: 5-14 hours
echo 📊 Processing: Complete dataset
echo ⚠️  This will take a VERY long time!
echo.
set /p confirm="Are you sure you want to continue? (y/N): "
if /i not "%confirm%"=="y" goto menu

echo Starting full retraining...
python scripts\monthly_full_retraining.py
if %errorlevel% equ 0 (
    echo ✅ Monthly retraining completed successfully!
) else (
    echo ❌ Monthly retraining failed. Check logs for details.
)
goto end

:test
echo.
echo ========================================
echo Starting Test Incremental Training
echo ========================================
echo ⏱️ Expected duration: 5-15 minutes
echo 📊 Processing: Small test dataset
echo.
echo Starting test training...
cd ..
python src\train_sku_nn_predictor_pytorch_optimized.py --mode incremental --days 1
if %errorlevel% equ 0 (
    echo ✅ Test training completed successfully!
) else (
    echo ❌ Test training failed. Check logs for details.
)
goto end

:end
echo.
echo ========================================
echo Training Process Complete
echo ========================================
echo.
echo 📋 Check the following for results:
echo    - logs\*.log files for detailed logs
echo    - models\sku_nn\ for updated models
echo    - data\fixacar_history.db for updated data
echo.
pause
goto menu

:exit
echo.
echo Goodbye!
exit /b 0
