@echo off
echo ========================================
echo Building Fixacar SKU Finder Executable
echo ========================================

echo.
echo Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)

echo.
echo Installing/Updating PyInstaller...
pip install pyinstaller

echo.
echo Cleaning previous builds...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

echo.
echo Building executable...
pyinstaller Fixacar_SKU_Finder.spec

echo.
if exist "dist\Fixacar_SKU_Finder.exe" (
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Executable created at: dist\Fixacar_SKU_Finder.exe
    echo.
    echo You can now distribute the entire 'dist' folder to your client.
    echo The client only needs to double-click 'Fixacar_SKU_Finder.exe' to run the application.
    echo.
) else (
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo.
    echo Please check the output above for errors.
    echo.
)

echo Press any key to exit...
pause > nul
