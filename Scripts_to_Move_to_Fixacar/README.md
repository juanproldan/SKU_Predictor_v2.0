# Fixacar SKU Finder Build & Deployment Scripts

This folder contains PowerShell scripts to automate building, packaging, and retraining the Fixacar SKU Finder application.

## Prerequisites
- Python and all required packages installed (see `requirements.txt` in the project root)
- PyInstaller installed (`pip install pyinstaller`)
- Run these scripts from the project root directory

## Scripts

### 1. Build Executables
**Script:** `build_executables.ps1`

- Builds the main GUI app and the model training workflow as Windows executables using PyInstaller.
- Output executables will be placed in the `dist` folder.

**Usage:**
```powershell
./build_executables.ps1
```

---

### 2. Package for Deployment
**Script:** `package_deployment.ps1`

- Packages the executables, models, data, and support files into a `Fixacar_Deploy` folder for easy distribution.

**Usage:**
```powershell
./package_deployment.ps1
```

---

### 3. Retrain the Model (Optional)
**Script:** `retrain_model.ps1`

- Runs the model training workflow to retrain the optimized SKU NN model.
- Use this after updating training data.

**Usage:**
```powershell
./retrain_model.ps1
```

---

## Deployment Checklist

Before packaging or moving the app to another computer, make sure the following files and folders are present and up to date:

- [ ] `dist/Fixacar_SKU_Finder.exe` (main app executable)
- [ ] `dist/Fixacar_SKU_Trainer.exe` (model training executable)
- [ ] `models/` folder (all VIN and SKU NN model files, encoders, tokenizers, including `models/sku_nn/`)
- [ ] `data/Maestro.xlsx` (user-confirmed SKUs, created/updated by the app)
- [ ] `data/fixacar_history.db` (SQLite database, used for search and training)
- [ ] `Source_Files/Equivalencias.xlsx` (synonym mapping for part descriptions)
- [ ] Any other required files in `Source_Files/` (e.g., `Consolidado.json` for retraining)
- [ ] `requirements.txt` (for reference or if you need to rebuild the executables)

**Tip:** The `package_deployment.ps1` script will collect most of these automatically, but always double-check this list before distributing.

---

## Notes
- Make sure all required files (models, data, etc.) are present before building or packaging.
- For troubleshooting, check the console output for error messages.
- You can edit these scripts to customize paths or add extra steps as needed.

---

For further help, see the main project README or contact the developer.
