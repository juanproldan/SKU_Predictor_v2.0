# Fixacar SKU Predictor - Client Deployment Package

This folder contains the complete client deployment package for the Fixacar SKU Predictor system.

## 📁 Folder Structure

```
Fixacar_SKU_Predictor_CLIENT/
├── Fixacar_SKU_Predictor.exe         # Main SKU prediction application (will be here when built)
├── download_consolidado.exe          # Automated data download utility (will be here when built)
├── train_vin_predictor.exe           # VIN model training executable (will be here when built)
├── train_sku_nn_predictor.exe        # SKU neural network training executable (will be here when built)
├── unified_consolidado_processor.exe # Data processing executable (will be here when built)
├── Source_Files/          # Data and configuration files
│   ├── Text_Processing_Rules.xlsx    # Text processing rules (IMPORTANT - manually maintained)
│   ├── Maestro.xlsx                  # Expert SKU database (IMPORTANT - manually maintained)
│   ├── Consolidado.json              # Latest automotive parts data (auto-downloaded)
│   └── processed_consolidado.db      # Processed database (auto-generated)
├── models/                # Trained AI models (will contain .joblib and .pth files when trained)
│   └── sku_nn/            # SKU neural network models directory
└── logs/                  # Application logs
    └── *.log              # Daily log files
```

## 🚀 Usage Instructions

### For End Users:
1. **Run SKU Predictor**: Double-click `Fixacar_SKU_Predictor.exe`
2. **Update Data**: The system automatically downloads latest data weekly

### For System Administrators:
1. **Manual Data Update**: Run `download_consolidado.exe`
2. **Model Retraining**: Run training utilities when needed:
   - `train_vin_predictor.exe` - Train VIN prediction models
   - `train_sku_nn_predictor.exe` - Train SKU neural network models
   - `unified_consolidado_processor.exe` - Process new data files
3. **Logs**: Check `logs/` folder for troubleshooting

## ⚠️ Important Files

**DO NOT DELETE OR MODIFY:**
- `Source_Files/Text_Processing_Rules.xlsx` - Contains manually curated text processing rules
- `Source_Files/Maestro.xlsx` - Contains expert-validated SKU database

These files contain important manually maintained data that cannot be easily recreated.

## 🔄 Automatic Updates

- **Data**: Consolidado.json is automatically downloaded weekly
- **Models**: Retrained automatically based on new data
- **Logs**: Rotated daily to prevent disk space issues

## 📞 Support

For technical support or issues, contact the development team with:
- Log files from the `logs/` folder
- Description of the issue
- Steps to reproduce the problem
