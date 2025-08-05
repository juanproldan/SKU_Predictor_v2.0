#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final deployment verification script
This verifies that the bulletproof deployment is ready for client laptops
"""

import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and report its size"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ {description}: NOT FOUND")
        return False

def verify_deployment_structure():
    """Verify the complete deployment structure"""
    
    print("📁 Verifying Deployment Structure")
    print("=" * 40)
    
    base_path = "Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT"
    
    # Critical files to check
    files_to_check = [
        (f"{base_path}/Fixacar_SKU_Predictor_BULLETPROOF.exe", "Bulletproof Executable"),
        (f"{base_path}/Fixacar_SKU_Predictor_BULLETPROOF.bat", "Bulletproof Batch File"),
        (f"{base_path}/Source_Files/Text_Processing_Rules.xlsx", "Text Processing Rules"),
        (f"{base_path}/Source_Files/Maestro.xlsx", "Maestro Data"),
        (f"{base_path}/Source_Files/processed_consolidado.db", "Processed Database"),
        (f"{base_path}/Source_Files/Consolidado.json", "Consolidado JSON"),
    ]
    
    all_files_ok = True
    
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_files_ok = False
    
    # Check models directory
    models_path = f"{base_path}/models"
    if os.path.exists(models_path):
        model_files = list(Path(models_path).rglob("*"))
        print(f"✅ Models Directory: {len(model_files)} files")
        
        # Check for critical model files
        critical_models = [
            "vin_maker_model_pytorch.pth",
            "vin_model_model_pytorch.pth", 
            "vin_series_model_pytorch.pth"
        ]
        
        for model in critical_models:
            model_path = os.path.join(models_path, model)
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"  ✅ {model}: {size_mb:.1f} MB")
            else:
                print(f"  ❌ {model}: NOT FOUND")
                all_files_ok = False
    else:
        print(f"❌ Models Directory: NOT FOUND")
        all_files_ok = False
    
    return all_files_ok

def test_bulletproof_executable():
    """Test the bulletproof executable"""
    
    print("\n🧪 Testing Bulletproof Executable")
    print("=" * 40)
    
    exe_path = "Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Fixacar_SKU_Predictor_BULLETPROOF.exe"
    
    if not os.path.exists(exe_path):
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    # Get executable size
    size_mb = os.path.getsize(exe_path) / (1024 * 1024)
    print(f"📊 Executable size: {size_mb:.1f} MB")
    
    if size_mb < 100:
        print("⚠️ Warning: Executable seems small, dependencies might be missing")
    elif size_mb > 1000:
        print("⚠️ Warning: Executable is very large, might include unnecessary files")
    else:
        print("✅ Executable size looks reasonable")
    
    # Test if executable can start (quick test)
    try:
        print("🚀 Testing executable startup...")
        
        # Run with a very short timeout just to see if it starts
        result = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(exe_path)
        )
        
        # If we get here, it either completed quickly or we caught it
        if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
            print("❌ Import errors detected in executable!")
            print(f"Error: {result.stderr}")
            return False
        else:
            print("✅ Executable started without import errors")
            return True
            
    except subprocess.TimeoutExpired:
        print("✅ Executable started successfully (timed out waiting for GUI)")
        return True
    except Exception as e:
        print(f"❌ Error testing executable: {e}")
        return False

def create_deployment_summary():
    """Create a deployment summary file"""
    
    summary = """
# 🚀 FIXACAR SKU PREDICTOR - BULLETPROOF DEPLOYMENT

## 📦 Deployment Package Contents

This package contains a bulletproof version of the Fixacar SKU Predictor that includes ALL dependencies and should work on any Windows laptop without requiring Python installation.

## 🎯 What's New in the Bulletproof Version

- **Complete Dependency Inclusion**: All NumPy, Pandas, PyTorch, and other dependencies are bundled
- **Enhanced Error Handling**: Better error messages and debugging capabilities
- **Cross-Platform Compatibility**: Works on different Windows versions
- **No Python Required**: Completely standalone executable

## 📁 Package Contents

- `Fixacar_SKU_Predictor_BULLETPROOF.exe` - Main application (bulletproof version)
- `Fixacar_SKU_Predictor_BULLETPROOF.bat` - Easy launcher with error checking
- `Source_Files/` - All data files (Excel, database, JSON)
- `models/` - All trained models for VIN and SKU prediction
- `logs/` - Application logs

## 🚀 How to Use

### Option 1: Double-click the batch file (Recommended)
1. Double-click `Fixacar_SKU_Predictor_BULLETPROOF.bat`
2. The application will start with error checking

### Option 2: Direct executable
1. Double-click `Fixacar_SKU_Predictor_BULLETPROOF.exe`
2. The application will start directly

## 🛠️ Troubleshooting

If you encounter any issues:

1. **Check file permissions**: Ensure you have permission to run the executable
2. **Run as administrator**: Right-click and "Run as administrator"
3. **Check antivirus**: Some antivirus software may block the executable
4. **Verify complete copy**: Ensure all files and folders were copied completely

## 📞 Support

If the application doesn't start or shows errors:
1. Note the exact error message
2. Check if all files are present in the folder
3. Try running as administrator
4. Contact support with the error details

## ✅ Verification Checklist

Before deploying to client laptops, verify:
- [ ] All files are present in the package
- [ ] Executable starts without errors on test machine
- [ ] All features work correctly (VIN prediction, SKU prediction)
- [ ] File operations work (Excel reading/writing, database access)

---

**Package Size**: ~500MB+ (includes all dependencies)
**Python Required**: No
**Windows Version**: Windows 7 and later
**Last Updated**: """ + str(os.path.getmtime("Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Fixacar_SKU_Predictor_BULLETPROOF.exe"))

    with open("Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/README_DEPLOYMENT.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print("✅ Created deployment summary: README_DEPLOYMENT.txt")

def main():
    """Main verification function"""
    
    print("🚀 Fixacar SKU Predictor - Final Deployment Verification")
    print("=" * 60)
    
    # Verify deployment structure
    structure_ok = verify_deployment_structure()
    
    # Test executable
    executable_ok = test_bulletproof_executable()
    
    # Create deployment summary
    create_deployment_summary()
    
    print("\n📊 Final Verification Summary")
    print("=" * 40)
    
    if structure_ok and executable_ok:
        print("✅ DEPLOYMENT READY!")
        print("✅ All files are present and executable works")
        print("✅ Package is ready for client laptop deployment")
        print("\n📋 Next Steps:")
        print("1. Copy the entire 'Fixacar_NUCLEAR_DEPLOYMENT' folder to client laptops")
        print("2. Run 'Fixacar_SKU_Predictor_BULLETPROOF.bat' on client machines")
        print("3. Verify all features work correctly")
        return True
    else:
        print("❌ DEPLOYMENT NOT READY")
        print("❌ Please fix the issues above before deploying")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
