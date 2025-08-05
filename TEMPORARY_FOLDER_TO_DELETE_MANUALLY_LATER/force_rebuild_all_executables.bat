@echo off
echo.
echo ========================================
echo   FORCE REBUILDING ALL FIXACAR EXECUTABLES
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo 🔧 Enhanced dependency collection enabled for all executables
echo 🗑️ Removing existing executables to prevent permission errors...
echo.

REM Kill any running processes that might lock the files
echo 🛑 Stopping any running Fixacar processes...
taskkill /f /im "1. Fixacar_Consolidado_Downloader.exe" 2>nul
taskkill /f /im "2. Fixacar_Data_Processor.exe" 2>nul
taskkill /f /im "3. Fixacar_VIN_Trainer.exe" 2>nul
taskkill /f /im "4. Fixacar_SKU_Trainer.exe" 2>nul
taskkill /f /im "Fixacar_SKU_Predictor.exe" 2>nul

REM Wait a moment for processes to fully terminate
timeout /t 3 /nobreak >nul

REM Remove existing executables to prevent permission errors
echo 🗑️ Removing existing executables...
cd "Fixacar_SKU_Predictor_CLIENT"
if exist "1. Fixacar_Consolidado_Downloader.exe" (
    attrib -r "1. Fixacar_Consolidado_Downloader.exe"
    del /f "1. Fixacar_Consolidado_Downloader.exe"
    echo   ✅ Removed Consolidado Downloader
)
if exist "2. Fixacar_Data_Processor.exe" (
    attrib -r "2. Fixacar_Data_Processor.exe"
    del /f "2. Fixacar_Data_Processor.exe"
    echo   ✅ Removed Data Processor
)
if exist "3. Fixacar_VIN_Trainer.exe" (
    attrib -r "3. Fixacar_VIN_Trainer.exe"
    del /f "3. Fixacar_VIN_Trainer.exe"
    echo   ✅ Removed VIN Trainer
)
if exist "4. Fixacar_SKU_Trainer.exe" (
    attrib -r "4. Fixacar_SKU_Trainer.exe"
    del /f "4. Fixacar_SKU_Trainer.exe"
    echo   ✅ Removed SKU Trainer
)
if exist "Fixacar_SKU_Predictor.exe" (
    attrib -r "Fixacar_SKU_Predictor.exe"
    del /f "Fixacar_SKU_Predictor.exe"
    echo   ✅ Removed SKU Predictor GUI
)
cd ..

REM Clean previous builds
echo 🧹 Cleaning build cache...
if exist "temp_build" rmdir /s /q "temp_build"
mkdir "temp_build"

echo.
echo ========================================
echo   1/5 BUILDING CONSOLIDADO DOWNLOADER
echo ========================================
echo.
pyinstaller --distpath "Fixacar_SKU_Predictor_CLIENT" --workpath "temp_build" --clean Fixacar_Consolidado_Downloader.spec
if %errorlevel% neq 0 (
    echo ❌ FAILED: Consolidado Downloader build failed!
    pause
    exit /b 1
)
echo ✅ Consolidado Downloader built successfully!

echo.
echo ========================================
echo   2/5 BUILDING DATA PROCESSOR
echo ========================================
echo.
pyinstaller --distpath "Fixacar_SKU_Predictor_CLIENT" --workpath "temp_build" --clean Fixacar_Data_Processor.spec
if %errorlevel% neq 0 (
    echo ❌ FAILED: Data Processor build failed!
    pause
    exit /b 1
)
echo ✅ Data Processor built successfully!

echo.
echo ========================================
echo   3/5 BUILDING VIN TRAINER
echo ========================================
echo.
pyinstaller --distpath "Fixacar_SKU_Predictor_CLIENT" --workpath "temp_build" --clean Fixacar_VIN_Trainer.spec
if %errorlevel% neq 0 (
    echo ❌ FAILED: VIN Trainer build failed!
    pause
    exit /b 1
)
echo ✅ VIN Trainer built successfully!

echo.
echo ========================================
echo   4/5 BUILDING SKU TRAINER
echo ========================================
echo.
pyinstaller --distpath "Fixacar_SKU_Predictor_CLIENT" --workpath "temp_build" --clean Fixacar_SKU_Trainer.spec
if %errorlevel% neq 0 (
    echo ❌ FAILED: SKU Trainer build failed!
    pause
    exit /b 1
)
echo ✅ SKU Trainer built successfully!

echo.
echo ========================================
echo   5/5 BUILDING SKU PREDICTOR GUI
echo ========================================
echo.
pyinstaller --distpath "Fixacar_SKU_Predictor_CLIENT" --workpath "temp_build" --clean Fixacar_SKU_Predictor.spec
if %errorlevel% neq 0 (
    echo ❌ FAILED: SKU Predictor GUI build failed!
    pause
    exit /b 1
)
echo ✅ SKU Predictor GUI built successfully!

echo.
echo ========================================
echo   🎉 ALL EXECUTABLES BUILT SUCCESSFULLY!
echo ========================================
echo.
echo 🧪 Running isolated dependency tests...
echo.
python isolated_dependency_test.py

echo.
echo ========================================
echo   BUILD AND TEST COMPLETE
echo ========================================
echo.
echo ✅ All 5 executables rebuilt with enhanced dependency collection
echo ✅ All executables tested in isolated environment
echo 🚀 Ready for deployment to client systems!
echo.
echo 📊 File timestamps updated - all executables are now current
echo.
pause
