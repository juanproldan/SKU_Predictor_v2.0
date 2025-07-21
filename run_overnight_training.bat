@echo off
echo ========================================
echo    SKU Neural Network Overnight Training
echo ========================================
echo.
echo Starting full training with optimized parameters...
echo This will run overnight and may take 8-12 hours.
echo.
echo Training Configuration:
echo - Mode: Full dataset (365K+ records)
echo - Batch Size: 256 (optimized)
echo - Max Epochs: 100
echo - Early Stopping: 20 epochs patience
echo - Learning Rate Scheduler: Enabled
echo.
echo Press Ctrl+C to stop training at any time.
echo.
pause

echo Starting training at %date% %time%
python src\train_sku_nn_predictor_pytorch_optimized.py --mode full

echo.
echo Training completed at %date% %time%
echo Check the models/sku_nn/ directory for the trained model.
echo.
pause
