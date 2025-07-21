@echo off
echo ========================================
echo Building SKU Predictor Executables
echo ========================================

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo.
echo Building executables...
echo.

REM 1. Build SKU Predictor GUI
echo 1. Building SKU Predictor GUI...
pyinstaller --onefile --windowed --name "Fixacar_SKU_Predictor" ^
    --add-data "Source_Files;Source_Files" ^
    --add-data "data;data" ^
    --add-data "models;models" ^
    --hidden-import torch ^
    --hidden-import sklearn ^
    --hidden-import pandas ^
    --hidden-import openpyxl ^
    --hidden-import joblib ^
    --hidden-import sqlite3 ^
    --hidden-import tkinter ^
    --hidden-import tkinter.ttk ^
    --hidden-import tkinter.messagebox ^
    --hidden-import tkinter.filedialog ^
    --icon=NONE ^
    src/main_app.py

if %errorlevel% equ 0 (
    echo    ✅ SKU Predictor GUI built successfully
) else (
    echo    ❌ Failed to build SKU Predictor GUI
)

echo.
echo 2. Building VIN Trainer...
pyinstaller --onefile --console --name "Fixacar_VIN_Trainer" ^
    --add-data "Source_Files;Source_Files" ^
    --add-data "data;data" ^
    --add-data "models;models" ^
    --hidden-import torch ^
    --hidden-import sklearn ^
    --hidden-import pandas ^
    --hidden-import sqlite3 ^
    --hidden-import joblib ^
    src/train_vin_predictor.py

if %errorlevel% equ 0 (
    echo    ✅ VIN Trainer built successfully
) else (
    echo    ❌ Failed to build VIN Trainer
)

echo.
echo 3. Building SKU Trainer...
pyinstaller --onefile --console --name "Fixacar_SKU_Trainer" ^
    --add-data "Source_Files;Source_Files" ^
    --add-data "data;data" ^
    --add-data "models;models" ^
    --hidden-import torch ^
    --hidden-import sklearn ^
    --hidden-import pandas ^
    --hidden-import sqlite3 ^
    --hidden-import joblib ^
    --hidden-import numpy ^
    src/train_sku_nn_predictor_pytorch_optimized.py

if %errorlevel% equ 0 (
    echo    ✅ SKU Trainer built successfully
) else (
    echo    ❌ Failed to build SKU Trainer
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executables created in dist/ folder:
echo - Fixacar_SKU_Predictor.exe (Main GUI)
echo - Fixacar_VIN_Trainer.exe (Weekly training)
echo - Fixacar_SKU_Trainer.exe (Monthly training)
echo.
echo 📋 Next Steps:
echo 1. Test each executable locally
echo 2. Copy dist/ folder to client laptop
echo 3. Manually schedule trainers in Windows Task Scheduler
echo.
pause
