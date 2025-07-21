@echo off
REM Setup Training Schedule for SKU Predictor
REM This script sets up Windows Task Scheduler for automated training

echo ========================================
echo SKU Predictor Training Schedule Setup
echo ========================================

REM Get current directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

echo Project Directory: %PROJECT_DIR%

REM Create weekly incremental training task
echo.
echo Setting up WEEKLY incremental training...
echo - Runs every Sunday at 2:00 AM
echo - Duration: 20-60 minutes

schtasks /create /tn "SKU_Predictor_Weekly_Training" /tr "python \"%PROJECT_DIR%\scripts\weekly_incremental_training.py\"" /sc weekly /d SUN /st 02:00 /f

if %errorlevel% equ 0 (
    echo ✅ Weekly training task created successfully
) else (
    echo ❌ Failed to create weekly training task
)

REM Create monthly full retraining task
echo.
echo Setting up MONTHLY full retraining...
echo - Runs first Saturday of each month at 10:00 PM
echo - Duration: 5-14 hours

schtasks /create /tn "SKU_Predictor_Monthly_Training" /tr "python \"%PROJECT_DIR%\scripts\monthly_full_retraining.py\"" /sc monthly /mo first /d SAT /st 22:00 /f

if %errorlevel% equ 0 (
    echo ✅ Monthly training task created successfully
) else (
    echo ❌ Failed to create monthly training task
)

echo.
echo ========================================
echo Schedule Setup Complete!
echo ========================================
echo.
echo 📅 WEEKLY TRAINING:
echo    - Every Sunday at 2:00 AM
echo    - Incremental updates (20-60 min)
echo    - Processes last 7 days of data
echo.
echo 📅 MONTHLY TRAINING:
echo    - First Saturday of month at 10:00 PM
echo    - Full retraining (5-14 hours)
echo    - Processes complete dataset
echo.
echo 📋 To view scheduled tasks:
echo    schtasks /query /tn "SKU_Predictor_*"
echo.
echo 🗑️ To remove scheduled tasks:
echo    schtasks /delete /tn "SKU_Predictor_Weekly_Training" /f
echo    schtasks /delete /tn "SKU_Predictor_Monthly_Training" /f
echo.
echo ⚠️  IMPORTANT NOTES:
echo    - Make sure Python is in your system PATH
echo    - Ensure Consolidado.json is updated before training
echo    - Check logs in the 'logs' folder for training status
echo    - Monthly training requires 5-14 hours of uninterrupted time
echo.

pause
