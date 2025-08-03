@echo off
REM SKU Predictor Logging Configuration Batch File
REM This provides an easy way for Windows users to set logging levels

echo ============================================================
echo SKU Predictor Logging Configuration
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if we have arguments
if "%1"=="" (
    echo Current logging configuration:
    echo.
    python set_logging_level.py --show
    echo.
    echo Available options:
    echo   set_logging.bat NORMAL    - Default production mode
    echo   set_logging.bat VERBOSE   - Detailed logging
    echo   set_logging.bat DEBUG     - Full debugging
    echo   set_logging.bat MINIMAL   - Errors only
    echo   set_logging.bat SILENT    - Critical errors only
    echo.
    echo   set_logging.bat show      - Show current configuration
    echo   set_logging.bat list      - List all available levels
    echo   set_logging.bat test      - Test current configuration
    echo.
    pause
    exit /b 0
)

REM Handle special commands
if /i "%1"=="show" (
    python set_logging_level.py --show
    pause
    exit /b 0
)

if /i "%1"=="list" (
    python set_logging_level.py --list
    pause
    exit /b 0
)

if /i "%1"=="test" (
    python set_logging_level.py --test
    pause
    exit /b 0
)

REM Set logging level
echo Setting logging level to: %1
echo.
python set_logging_level.py --level %1

if errorlevel 1 (
    echo.
    echo ERROR: Failed to set logging level
    echo Valid levels are: SILENT, MINIMAL, NORMAL, VERBOSE, DEBUG
) else (
    echo.
    echo SUCCESS: Logging level set to %1
    echo.
    echo IMPORTANT: Restart any running applications to apply changes
)

echo.
pause
