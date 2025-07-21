@echo off 
cd /d "%~dp0" 
echo Starting VIN Training at %date% %time% 
python src\train_vin_predictor.py 
echo VIN Training completed at %date% %time% 
pause 
