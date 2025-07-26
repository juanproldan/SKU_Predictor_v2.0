@echo off
echo ========================================
echo Building Fixacar SKU Predictor Executables
echo IMPROVED VERSION - Better Dependency Handling
echo ========================================

REM Set error handling
setlocal enabledelayedexpansion

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Check if virtual environment is active (recommended)
python -c "import sys; print('Virtual env active' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'No virtual env')"

echo.
echo 📦 Checking and installing required packages...

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo ❌ ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

REM Check critical dependencies
echo Checking critical dependencies...
python -c "import numpy, pandas, torch, sklearn, joblib, openpyxl, fuzzywuzzy, Levenshtein" 2>nul
if %errorlevel% neq 0 (
    echo ❌ ERROR: Missing critical dependencies
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✅ All dependencies found

REM Clean previous builds
echo.
echo 🧹 Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "*.spec" (
    echo Found existing spec files - keeping them
) else (
    echo ❌ ERROR: Spec files not found. Please ensure .spec files are in the current directory.
    pause
    exit /b 1
)

echo.
echo 🔨 Building executables using spec files...
echo.

REM 1. Build SKU Predictor GUI using spec file
echo 1. Building SKU Predictor GUI...
pyinstaller --clean deployment/fixacar_sku_predictor.spec

if %errorlevel% equ 0 (
    echo    ✅ SKU Predictor GUI built successfully
    set "gui_success=1"
) else (
    echo    ❌ Failed to build SKU Predictor GUI
    set "gui_success=0"
)

echo.
echo 2. Building SKU Trainer...
pyinstaller --clean deployment/Fixacar_SKU_Trainer.spec

if %errorlevel% equ 0 (
    echo    ✅ SKU Trainer built successfully
    set "sku_success=1"
) else (
    echo    ❌ Failed to build SKU Trainer
    set "sku_success=0"
)

echo.
echo ========================================
echo Build Summary
echo ========================================

if "%gui_success%"=="1" (
    echo ✅ SKU Predictor GUI: SUCCESS
) else (
    echo ❌ SKU Predictor GUI: FAILED
)

if "%vin_success%"=="1" (
    echo ✅ VIN Trainer: SUCCESS
) else (
    echo ❌ VIN Trainer: FAILED
)

if "%sku_success%"=="1" (
    echo ✅ SKU Trainer: SUCCESS
) else (
    echo ❌ SKU Trainer: FAILED
)

echo.
if exist "dist\Fixacar_SKU_Predictor.exe" (
    echo 📁 Executables created in dist\ folder:
    dir /b dist\*.exe
    echo.
    echo 📋 Next Steps:
    echo 1. Test each executable locally BEFORE sending to client
    echo 2. Run: dist\Fixacar_SKU_Predictor.exe
    echo 3. Verify all features work correctly
    echo 4. Copy entire dist\ folder to client laptop
    echo 5. Set up Windows Task Scheduler for trainers
    echo.
    echo ⚠️  IMPORTANT: Test locally first to avoid client-side errors!
) else (
    echo ❌ No executables were created successfully
    echo Please check the error messages above and fix issues before retrying.
)

echo.
pause
