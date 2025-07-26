@echo off
echo.
echo ============================================================
echo 📅 FIXACAR CONSOLIDADO DOWNLOADER - SCHEDULER SETUP
echo ============================================================
echo.

REM Configuration
set TASK_NAME=Fixacar_Consolidado_Download
set EXE_PATH=%~dp0dist\Fixacar_Consolidado_Downloader.exe
set LOG_PATH=%~dp0logs\scheduler.log

echo 🔧 Setting up automated Consolidado.json download...
echo.
echo 📁 Executable: %EXE_PATH%
echo 📋 Task Name: %TASK_NAME%
echo 📝 Log Path: %LOG_PATH%
echo.

REM Check if executable exists
if not exist "%EXE_PATH%" (
    echo ❌ ERROR: Executable not found at %EXE_PATH%
    echo Please build the executable first using build_executables.bat
    pause
    exit /b 1
)

REM Create logs directory
if not exist "%~dp0logs" mkdir "%~dp0logs"

echo 🗑️ Removing existing task (if any)...
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

echo.
echo 📅 Creating scheduled task...
echo    - Runs daily at 6:00 AM
echo    - Downloads latest Consolidado.json
echo    - Logs all activity
echo.

REM Create the scheduled task
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "\"%EXE_PATH%\"" ^
    /sc daily ^
    /st 06:00 ^
    /ru "SYSTEM" ^
    /rl highest ^
    /f

if %ERRORLEVEL% EQU 0 (
    echo ✅ Task created successfully!
    echo.
    echo 📋 Task Details:
    echo    - Name: %TASK_NAME%
    echo    - Schedule: Daily at 6:00 AM
    echo    - Action: Download Consolidado.json
    echo    - User: SYSTEM (runs even when logged out)
    echo.
    echo 🔍 To view the task:
    echo    taskschd.msc
    echo.
    echo 🧪 To test the task manually:
    echo    schtasks /run /tn "%TASK_NAME%"
    echo.
    echo 🗑️ To remove the task:
    echo    schtasks /delete /tn "%TASK_NAME%" /f
    echo.
) else (
    echo ❌ ERROR: Failed to create scheduled task
    echo Please run this script as Administrator
    echo.
)

echo ============================================================
echo 📋 SETUP COMPLETE
echo ============================================================
echo.
echo The Consolidado.json file will be automatically downloaded
echo daily at 6:00 AM to keep your data up to date.
echo.

pause
