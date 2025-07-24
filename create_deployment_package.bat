@echo off
echo.
echo ============================================================
echo 📦 FIXACAR SKU PREDICTOR - DEPLOYMENT PACKAGE CREATOR
echo ============================================================
echo.

set DEPLOYMENT_DIR=Fixacar_Deployment_Package
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo 🗂️  Creating deployment package: %DEPLOYMENT_DIR%
echo 📅 Timestamp: %TIMESTAMP%
echo.

REM Clean up any existing deployment package
if exist "%DEPLOYMENT_DIR%" (
    echo 🧹 Cleaning up existing deployment package...
    rmdir /s /q "%DEPLOYMENT_DIR%"
)

REM Create main deployment directory
mkdir "%DEPLOYMENT_DIR%"

echo.
echo ============================================================
echo 📁 COPYING ESSENTIAL FILES
echo ============================================================

REM Copy executables
echo 🚀 Copying executables from dist/...
if exist "dist" (
    xcopy "dist" "%DEPLOYMENT_DIR%\dist" /E /I /Y
    echo    ✅ Executables copied
) else (
    echo    ❌ ERROR: dist/ folder not found! Run build_executables.bat first
    pause
    exit /b 1
)

REM Copy Source_Files
echo 📄 Copying Source_Files/...
if exist "Source_Files" (
    xcopy "Source_Files" "%DEPLOYMENT_DIR%\Source_Files" /E /I /Y
    echo    ✅ Source_Files copied
) else (
    echo    ⚠️  WARNING: Source_Files/ folder not found
)

REM Copy data folder
echo 💾 Copying data/...
if exist "data" (
    xcopy "data" "%DEPLOYMENT_DIR%\data" /E /I /Y
    echo    ✅ Data folder copied
) else (
    echo    ❌ ERROR: data/ folder not found! This is required for the app to work
    pause
    exit /b 1
)

REM Copy models folder
echo 🧠 Copying models/...
if exist "models" (
    xcopy "models" "%DEPLOYMENT_DIR%\models" /E /I /Y
    echo    ✅ Models folder copied
) else (
    echo    ❌ ERROR: models/ folder not found! This is required for predictions
    pause
    exit /b 1
)

REM Copy deployment documentation
echo 📋 Copying deployment documentation...
if exist "DEPLOYMENT_CHECKLIST.md" copy "DEPLOYMENT_CHECKLIST.md" "%DEPLOYMENT_DIR%\"
if exist "CLIENT_DEPLOYMENT_GUIDE.md" copy "CLIENT_DEPLOYMENT_GUIDE.md" "%DEPLOYMENT_DIR%\"

echo.
echo ============================================================
echo 📊 DEPLOYMENT PACKAGE SUMMARY
echo ============================================================

REM Calculate sizes
for /f %%i in ('dir "%DEPLOYMENT_DIR%\dist" /s /-c ^| find "bytes"') do set DIST_SIZE=%%i
for /f %%i in ('dir "%DEPLOYMENT_DIR%" /s /-c ^| find "bytes"') do set TOTAL_SIZE=%%i

echo 📁 Package Location: %DEPLOYMENT_DIR%\
echo 📊 Executables Size: %DIST_SIZE% bytes
echo 📊 Total Package Size: %TOTAL_SIZE% bytes
echo.

REM List contents
echo 📂 Package Contents:
dir "%DEPLOYMENT_DIR%" /B

echo.
echo ============================================================
echo ✅ DEPLOYMENT PACKAGE CREATED SUCCESSFULLY!
echo ============================================================
echo.
echo 🎯 Next Steps:
echo    1. Copy entire '%DEPLOYMENT_DIR%' folder to client laptop
echo    2. Place in C:\Fixacar\ (recommended)
echo    3. Run Fixacar_SKU_Predictor.exe to test
echo    4. Set up Windows Task Scheduler for automated training
echo.
echo 📋 See DEPLOYMENT_CHECKLIST.md for detailed instructions
echo.

pause
