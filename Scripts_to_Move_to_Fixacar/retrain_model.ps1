# retrain_model.ps1
# Runs the model training workflow to retrain the optimized SKU NN model.

Write-Host "=== Running SKU NN Model Training Workflow ==="

# Activate virtual environment if needed (uncomment and adjust if using venv)
# & "venv_tf\Scripts\Activate.ps1"

# Run the training script
python src\train_sku_nn_predictor_pytorch_optimized.py

Write-Host "=== Model Training Complete! ==="
Write-Host "Check the 'models/sku_nn/' directory for the new model file."
