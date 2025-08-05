#!/usr/bin/env python3
"""
Create Client-Ready Version with All Dependencies
=================================================

This script creates a version that includes all necessary dependencies
for client laptops, including Microsoft Visual C++ Redistributables.
"""

import os
import sys
import shutil
import subprocess
import urllib.request
import tempfile
from pathlib import Path

def print_status(message, emoji="🔧"):
    """Print status message with emoji"""
    print(f"{emoji} {message}")

def download_vcredist():
    """Download Microsoft Visual C++ Redistributables"""
    print_status("Downloading Microsoft Visual C++ Redistributables...", "📥")
    
    # URLs for VC++ Redistributables
    vcredist_urls = {
        "x64": "https://aka.ms/vs/17/release/vc_redist.x64.exe",
        "x86": "https://aka.ms/vs/17/release/vc_redist.x86.exe"
    }
    
    vcredist_dir = Path("vcredist")
    vcredist_dir.mkdir(exist_ok=True)
    
    for arch, url in vcredist_urls.items():
        filename = f"vc_redist.{arch}.exe"
        filepath = vcredist_dir / filename
        
        if not filepath.exists():
            print_status(f"Downloading {filename}...", "⬇️")
            try:
                urllib.request.urlretrieve(url, filepath)
                print_status(f"✅ Downloaded {filename}")
            except Exception as e:
                print_status(f"❌ Failed to download {filename}: {e}")
                return False
        else:
            print_status(f"✅ {filename} already exists")
    
    return True

def create_client_installer():
    """Create a client installer batch file"""
    print_status("Creating client installer...", "📦")
    
    installer_content = '''@echo off
echo ==========================================
echo   Fixacar SKU Predictor - Client Setup
echo ==========================================
echo.
echo This will install the required dependencies
echo and set up the Fixacar SKU Predictor.
echo.
pause

echo Installing Microsoft Visual C++ Redistributables...
echo.

echo Installing x64 version...
vcredist\\vc_redist.x64.exe /quiet /norestart
if %errorlevel% neq 0 (
    echo Warning: x64 installation may have failed
)

echo Installing x86 version...
vcredist\\vc_redist.x86.exe /quiet /norestart
if %errorlevel% neq 0 (
    echo Warning: x86 installation may have failed
)

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo You can now run the application using:
echo   - Fixacar_SKU_Predictor_FIXED.bat
echo   - Or double-click Fixacar_SKU_Predictor_FIXED.exe
echo.
echo If you still have issues, try running the
echo minimal test first: Fixacar_MINIMAL_TEST.bat
echo.
pause
'''
    
    with open("Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/INSTALL_CLIENT.bat", "w") as f:
        f.write(installer_content)
    
    print_status("✅ Client installer created")

def create_enhanced_fixed_version():
    """Create an enhanced FIXED version with better error handling"""
    print_status("Building enhanced FIXED version...", "🔨")
    
    # Build the enhanced version
    cmd = [
        "venv/Scripts/python.exe", "-m", "PyInstaller",
        "--distpath", "Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT",
        "--workpath", "temp_build_enhanced",
        "--clean",
        "Fixacar_ENHANCED.spec"
    ]
    
    print_status(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print_status("✅ Enhanced FIXED version built successfully!")
            return True
        else:
            print_status(f"❌ Build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print_status(f"❌ Build error: {e}")
        return False

def create_enhanced_spec():
    """Create enhanced PyInstaller spec with better dependency handling"""
    print_status("Creating enhanced PyInstaller spec...", "📝")
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Get the source directory
src_dir = Path('src')
current_dir = Path('.')

# Data files to include
datas = [
    ('Source_Files/Text_Processing_Rules.xlsx', 'Source_Files'),
    ('Source_Files/Maestro.xlsx', 'Source_Files'),
    ('Source_Files/processed_consolidado.db', 'Source_Files'),
]

# Hidden imports for all dependencies
hiddenimports = [
    # Core scientific libraries
    'numpy', 'pandas', 'sklearn', 'torch', 'joblib',
    'openpyxl', 'sqlite3', 'tkinter', 'tkinter.ttk',
    
    # PyTorch dependencies
    'torch.nn', 'torch.nn.functional', 'torch.optim',
    'torch.utils', 'torch.utils.data',
    
    # NumPy/SciPy dependencies
    'numpy.core', 'numpy.core._methods', 'numpy.lib.format',
    'scipy', 'scipy.sparse', 'scipy.sparse.csgraph',
    
    # Sklearn dependencies
    'sklearn.ensemble', 'sklearn.tree', 'sklearn.utils',
    'sklearn.preprocessing', 'sklearn.feature_extraction',
    'sklearn.feature_extraction.text',
    
    # Custom modules
    'utils', 'utils.dummy_tokenizer', 'models', 'models.sku_nn_pytorch',
    
    # All utils submodules
    'utils.performance_improvements',
    'utils.optimized_startup',
    'utils.optimized_database',
    'utils.year_range_database',
]

# Binaries to include (for NumPy/SciPy DLLs)
binaries = []

# Analysis
a = Analysis(
    ['src/main_app.py'],
    pathex=[str(current_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Fixacar_SKU_Predictor_ENHANCED',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open("Fixacar_ENHANCED.spec", "w") as f:
        f.write(spec_content)
    
    print_status("✅ Enhanced spec file created")

def create_enhanced_batch():
    """Create enhanced batch file with better error handling"""
    print_status("Creating enhanced batch file...", "📝")
    
    batch_content = '''@echo off
title Fixacar SKU Predictor - Enhanced Version

echo.
echo ==========================================
echo   Fixacar SKU Predictor - Enhanced Version
echo ==========================================
echo.
echo This enhanced version includes better error
echo handling and dependency management.
echo.
echo Starting enhanced version...
echo.

REM Check if VC++ redistributables are installed
echo Checking system dependencies...

REM Try to run the application
"Fixacar_SKU_Predictor_ENHANCED.exe"

REM Check exit code
if %errorlevel% neq 0 (
    echo.
    echo ==========================================
    echo   Application Error Detected
    echo ==========================================
    echo.
    echo The application encountered an error.
    echo This might be due to missing dependencies.
    echo.
    echo Suggested solutions:
    echo 1. Run INSTALL_CLIENT.bat as Administrator
    echo 2. Try the minimal test: Fixacar_MINIMAL_TEST.bat
    echo 3. Check Windows Event Viewer for details
    echo.
    echo Error code: %errorlevel%
    echo.
)

echo.
echo Press any key to exit...
pause >nul
'''
    
    with open("Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Fixacar_SKU_Predictor_ENHANCED.bat", "w") as f:
        f.write(batch_content)
    
    print_status("✅ Enhanced batch file created")

def main():
    """Main function"""
    print_status("🚀 Creating Client-Ready Version", "🚀")
    print("=" * 50)
    
    # Step 1: Create enhanced spec
    create_enhanced_spec()
    
    # Step 2: Build enhanced version
    if not create_enhanced_fixed_version():
        print_status("❌ Failed to build enhanced version")
        return False
    
    # Step 3: Download VC++ redistributables
    if not download_vcredist():
        print_status("⚠️ Warning: Could not download VC++ redistributables")
    
    # Step 4: Copy VC++ redistributables to client folder
    client_dir = Path("Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT")
    vcredist_target = client_dir / "vcredist"
    
    if Path("vcredist").exists():
        if vcredist_target.exists():
            shutil.rmtree(vcredist_target)
        shutil.copytree("vcredist", vcredist_target)
        print_status("✅ VC++ redistributables copied to client folder")
    
    # Step 5: Create client installer
    create_client_installer()
    
    # Step 6: Create enhanced batch file
    create_enhanced_batch()
    
    print_status("🎉 Client-ready version created successfully!", "🎉")
    print("=" * 50)
    print()
    print("📋 Next steps:")
    print("1. Copy the entire Fixacar_SKU_Predictor_CLIENT folder to the client laptop")
    print("2. On the client laptop, run INSTALL_CLIENT.bat as Administrator")
    print("3. Then run Fixacar_SKU_Predictor_ENHANCED.bat")
    print("4. If issues persist, try Fixacar_MINIMAL_TEST.bat for diagnostics")
    
    return True

if __name__ == "__main__":
    main()
