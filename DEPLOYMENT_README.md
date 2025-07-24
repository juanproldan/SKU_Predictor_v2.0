# 🚀 Fixacar SKU Predictor - Deployment Guide

## 📋 Overview
This guide provides step-by-step instructions for creating and deploying standalone executables for the Fixacar SKU Predictor system, ensuring **no dependency issues** on client machines.

## 🔧 Prerequisites

### Development Machine Setup
1. **Python 3.8+** installed
2. **Virtual environment** (recommended)
3. **All dependencies** installed: `pip install -r requirements.txt`
4. **PyInstaller** installed: `pip install pyinstaller`

### Required Files Structure
```
Project Root/
├── src/
│   ├── main_app.py
│   ├── train_vin_predictor.py
│   ├── train_sku_nn_predictor_pytorch_optimized.py
│   └── utils/
├── data/
├── models/
├── Source_Files/
├── requirements.txt
├── *.spec files
└── build scripts
```

## 🛠️ Building Executables

### Step 1: Verify Dependencies
```bash
python verify_dependencies.py
```
This script checks:
- ✅ Python version compatibility
- ✅ All required packages installed
- ✅ Custom modules can be imported
- ✅ Data files exist
- ✅ PyInstaller is ready

**Fix any issues before proceeding!**

### Step 2: Build Executables
```bash
# Use the improved build script
build_executables_improved.bat
```

This creates:
- `Fixacar_SKU_Predictor.exe` - Main GUI application
- `Fixacar_VIN_Trainer.exe` - Weekly VIN training
- `Fixacar_SKU_Trainer.exe` - Monthly SKU training

### Step 3: Test Executables
```bash
python test_executables.py
```
This verifies:
- ✅ Executables launch without errors
- ✅ Dependencies are properly bundled
- ✅ No immediate crashes
- ✅ Reasonable file sizes

## 🎯 Key Improvements Over Previous Version

### 1. **Comprehensive Dependency Handling**
- **Added missing numpy imports** (main cause of previous error)
- **Explicit PyTorch submodules** (torch.nn, torch.optim, etc.)
- **Complete sklearn submodules** (preprocessing, metrics, etc.)
- **Text processing libraries** (fuzzywuzzy, Levenshtein)

### 2. **Advanced PyInstaller Configuration**
- **Spec files** for precise control over builds
- **Collect-submodules** for complex packages
- **Data file inclusion** for all required resources
- **Hidden imports** for dynamic imports

### 3. **Robust Testing Framework**
- **Pre-build verification** of dependencies
- **Post-build testing** of executables
- **Size validation** to ensure dependencies are bundled
- **Launch testing** to catch immediate errors

### 4. **Better Error Handling**
- **Detailed error messages** during build process
- **Dependency checking** before building
- **Build status reporting** for each executable
- **Troubleshooting guidance**

## 📦 Deployment to Client

### Step 1: Prepare Deployment Package
1. **Copy entire `dist/` folder** to client machine
2. **Recommended location**: `C:\Fixacar\`
3. **Include data files**: Ensure `data/`, `models/`, `Source_Files/` are present

### Step 2: Test on Client Machine
1. **Run main application**: `Fixacar_SKU_Predictor.exe`
2. **Verify functionality**: Test with sample data
3. **Check for errors**: No import errors should occur

### Step 3: Setup Automation
Use **Windows Task Scheduler** to schedule:

#### Weekly VIN Training
- **Program**: `C:\Fixacar\Fixacar_VIN_Trainer.exe`
- **Schedule**: Weekly, Sunday 2:00 AM
- **Run as**: System account or admin user

#### Monthly SKU Training  
- **Program**: `C:\Fixacar\Fixacar_SKU_Trainer.exe`
- **Schedule**: Monthly, 1st Sunday 3:00 AM
- **Run as**: System account or admin user

## 🐛 Troubleshooting

### Common Issues and Solutions

#### "Failed to execute script" Error
- **Cause**: Missing dependencies or corrupted build
- **Solution**: Rebuild using improved script, verify all dependencies

#### "numpy import error" (Previous Issue)
- **Cause**: numpy not included in hidden imports
- **Solution**: ✅ **FIXED** in improved build script

#### "DLL load failed" Error
- **Cause**: Missing Visual C++ Redistributable
- **Solution**: Install Microsoft Visual C++ Redistributable on client

#### Slow Startup
- **Cause**: Normal behavior - dependencies loading
- **Solution**: Wait 10-30 seconds on first run

#### Large File Sizes
- **Cause**: ML libraries (PyTorch, sklearn) are large
- **Expected**: 200-800 MB per executable is normal

### Debug Mode
For detailed error information, run executables from command line:
```cmd
cd C:\Fixacar
Fixacar_SKU_Predictor.exe
```

## 📊 File Size Expectations

| Executable | Expected Size | Dependencies |
|------------|---------------|--------------|
| SKU Predictor GUI | 300-500 MB | PyTorch, sklearn, pandas, tkinter |
| VIN Trainer | 200-400 MB | PyTorch, sklearn, pandas |
| SKU Trainer | 300-500 MB | PyTorch, sklearn, pandas |

## 🔄 Maintenance

### Regular Updates
1. **Update source code** as needed
2. **Rebuild executables** using same process
3. **Test thoroughly** before deployment
4. **Replace client files** with new versions

### Data Updates
- **Maestro.xlsx**: Update in `Source_Files/`
- **Training data**: Automatic via scheduled trainers
- **Models**: Regenerated by trainers

## 📞 Support

### Before Contacting Support
1. ✅ Run `verify_dependencies.py`
2. ✅ Check build logs for errors
3. ✅ Test executables locally
4. ✅ Verify client machine requirements

### Contact Information
- **Developer**: [Your contact information]
- **Documentation**: This README and generated checklists
- **Logs**: Check console output and error messages

---

## 🎉 Success Criteria

Your deployment is successful when:
- ✅ All executables build without errors
- ✅ GUI launches and functions work
- ✅ Trainers can run without import errors
- ✅ Client can use application without issues
- ✅ Scheduled training works automatically

**Remember**: Always test locally before deploying to client!
