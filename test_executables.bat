@echo off
echo ========================================
echo Testing SKU Predictor Executables
echo ========================================

REM Check if executables exist
if not exist "dist\Fixacar_SKU_Predictor.exe" (
    echo ❌ Fixacar_SKU_Predictor.exe not found in dist folder
    echo Please run build_executables.bat first
    pause
    exit /b 1
)

if not exist "dist\Fixacar_VIN_Trainer.exe" (
    echo ❌ Fixacar_VIN_Trainer.exe not found in dist folder
    echo Please run build_executables.bat first
    pause
    exit /b 1
)

if not exist "dist\Fixacar_SKU_Trainer.exe" (
    echo ❌ Fixacar_SKU_Trainer.exe not found in dist folder
    echo Please run build_executables.bat first
    pause
    exit /b 1
)

echo ✅ All executables found in dist folder
echo.

echo Testing executables...
echo.

echo 1. Testing VIN Trainer (this may take a few minutes)...
echo Starting VIN Trainer test at %date% %time%
cd dist
Fixacar_VIN_Trainer.exe
cd ..
echo VIN Trainer test completed at %date% %time%
echo.

echo 2. Testing SKU Trainer (this may take longer)...
echo Starting SKU Trainer test at %date% %time%
cd dist
Fixacar_SKU_Trainer.exe
cd ..
echo SKU Trainer test completed at %date% %time%
echo.

echo 3. Testing SKU Predictor GUI...
echo Starting GUI application...
echo (GUI will open in a new window - test manually)
cd dist
start Fixacar_SKU_Predictor.exe
cd ..
echo.

echo ========================================
echo Test Summary
echo ========================================
echo.
echo ✅ VIN Trainer: Completed
echo ✅ SKU Trainer: Completed  
echo ✅ GUI Application: Launched
echo.
echo 📋 Manual GUI Test Checklist:
echo - [ ] Application opens without errors
echo - [ ] VIN input field works
echo - [ ] Part descriptions input works
echo - [ ] "Find SKUs" button works
echo - [ ] Predictions are displayed
echo - [ ] Save functionality works
echo.
echo If all tests pass, executables are ready for deployment!
echo.
pause
