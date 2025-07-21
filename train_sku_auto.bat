@echo off 
cd /d "%~dp0" 
echo Starting SKU Training at %date% %time% 
python src\train_sku_nn_predictor_pytorch_optimized.py 
echo SKU Training completed at %date% %time% 
pause 
