# Client Deployment Guide - SKU Predictor v2.0

## 🎯 **Overview**
This guide explains how to deploy the SKU Predictor system to a client laptop with automated training.

## 📦 **What You'll Deploy**

### **Executables:**
- `Fixacar_SKU_Predictor.exe` - Main GUI application
- `Fixacar_VIN_Trainer.exe` - Weekly VIN model training
- `Fixacar_SKU_Trainer.exe` - Monthly SKU model training

### **Required Files:**
- `Source_Files/` folder (Consolidado.json, Equivalencias.xlsx)
- `data/` folder (databases, Maestro.xlsx)
- `models/` folder (trained models)

## 🔧 **Step 1: Build Executables**

On your development machine:

```batch
# Run the build script
build_executables.bat
```

This creates:
- `dist/Fixacar_SKU_Predictor.exe`
- `dist/Fixacar_VIN_Trainer.exe` 
- `dist/Fixacar_SKU_Trainer.exe`

## 📁 **Step 2: Prepare Deployment Package**

Create a folder structure on client laptop:
```
C:\Fixacar_SKU_Predictor\
├── Fixacar_SKU_Predictor.exe
├── Fixacar_VIN_Trainer.exe
├── Fixacar_SKU_Trainer.exe
├── Source_Files\
│   ├── Consolidado.json
│   └── Equivalencias.xlsx
├── data\
│   ├── Maestro.xlsx
│   ├── consolidado.db
│   └── fixacar_history.db
└── models\
    ├── vin_*.joblib (VIN models)
    ├── vin_*.pth (PyTorch models)
    └── sku_nn\ (SKU neural network)
```

## ⏰ **Step 3: Schedule Automated Training**

### **Open Windows Task Scheduler:**
1. Press `Windows + R`
2. Type: `taskschd.msc`
3. Press Enter

### **Create VIN Training Task (Weekly):**
1. Click "Create Basic Task..."
2. **Name:** `Fixacar_VIN_Training`
3. **Trigger:** Weekly, Sunday, 2:00 AM
4. **Action:** Start a program
5. **Program:** `C:\Fixacar_SKU_Predictor\Fixacar_VIN_Trainer.exe`
6. **Start in:** `C:\Fixacar_SKU_Predictor`

### **Create SKU Training Task (Monthly):**
1. Click "Create Basic Task..."
2. **Name:** `Fixacar_SKU_Training`
3. **Trigger:** Monthly, 1st day, 3:00 AM
4. **Action:** Start a program
5. **Program:** `C:\Fixacar_SKU_Predictor\Fixacar_SKU_Trainer.exe`
6. **Start in:** `C:\Fixacar_SKU_Predictor`

## 🧪 **Step 4: Test Everything**

### **Test Main Application:**
1. Double-click `Fixacar_SKU_Predictor.exe`
2. Enter a VIN and part descriptions
3. Verify predictions work

### **Test Training (Manual):**
1. Right-click VIN training task → "Run"
2. Check for completion in Task Scheduler History
3. Repeat for SKU training task

## 📋 **Step 5: Client Instructions**

### **Daily Use:**
- Double-click `Fixacar_SKU_Predictor.exe` to start
- Application runs independently

### **Automated Training:**
- **Weekly:** VIN models update every Sunday at 2 AM
- **Monthly:** SKU models retrain on 1st of month at 3 AM
- **No user action required** - runs automatically

### **Troubleshooting:**
- Check Task Scheduler History for training status
- Executables create log files in same directory
- Contact support if issues persist

## 🔄 **Maintenance Schedule**

| Task | Frequency | Time | Duration |
|------|-----------|------|----------|
| VIN Training | Weekly (Sunday) | 2:00 AM | 15-30 min |
| SKU Training | Monthly (1st) | 3:00 AM | 1-2 hours |

## ✅ **Deployment Checklist**

- [ ] Executables built successfully
- [ ] All required files copied to client
- [ ] Main application tested
- [ ] VIN training task scheduled
- [ ] SKU training task scheduled
- [ ] Both training tasks tested manually
- [ ] Client trained on daily usage
- [ ] Support contact information provided

## 🚀 **Benefits of This Approach**

- **No Python installation** required on client
- **Self-contained executables** with all dependencies
- **Automated training** keeps models current
- **Simple scheduling** with Windows Task Scheduler
- **Easy maintenance** and troubleshooting
