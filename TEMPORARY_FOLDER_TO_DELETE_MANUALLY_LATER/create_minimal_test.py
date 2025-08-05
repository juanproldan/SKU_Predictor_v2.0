#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a minimal test version to identify where the application is getting stuck
"""

import os
import sys
import subprocess

def create_minimal_main():
    """Create a minimal version of main_app.py for testing"""
    
    minimal_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MINIMAL TEST VERSION - Fixacar SKU Predictor
This version will help identify exactly where the application is getting stuck
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import traceback

print("🚀 MINIMAL TEST VERSION STARTING...")
print("=" * 50)

def test_imports():
    """Test all critical imports step by step"""
    print("📦 Testing imports...")
    
    try:
        print("  Testing basic Python modules...")
        import json, sqlite3, pickle, datetime
        print("  ✅ Basic Python modules OK")
        
        print("  Testing NumPy...")
        import numpy as np
        print(f"  ✅ NumPy {np.__version__} OK")
        
        print("  Testing Pandas...")
        import pandas as pd
        print(f"  ✅ Pandas {pd.__version__} OK")
        
        print("  Testing PyTorch...")
        import torch
        print(f"  ✅ PyTorch {torch.__version__} OK")
        
        print("  Testing scikit-learn...")
        import sklearn
        print(f"  ✅ Scikit-learn {sklearn.__version__} OK")
        
        print("  Testing openpyxl...")
        import openpyxl
        print(f"  ✅ OpenPyXL {openpyxl.__version__} OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_custom_imports():
    """Test custom module imports"""
    print("🔧 Testing custom imports...")
    
    try:
        print("  Testing utils...")
        from utils.dummy_tokenizer import DummyTokenizer
        print("  ✅ utils.dummy_tokenizer OK")
        
        print("  Testing models...")
        from models.sku_nn_pytorch import load_model
        print("  ✅ models.sku_nn_pytorch OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Custom import failed: {e}")
        traceback.print_exc()
        return False

def test_file_access():
    """Test file access"""
    print("📁 Testing file access...")
    
    current_dir = os.getcwd()
    print(f"  Current directory: {current_dir}")
    
    # Check for critical files
    files_to_check = [
        "Source_Files/Text_Processing_Rules.xlsx",
        "Source_Files/Maestro.xlsx", 
        "Source_Files/processed_consolidado.db"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)
            print(f"  ✅ {file_path}: {size:.1f} MB")
        else:
            print(f"  ⚠️ {file_path}: NOT FOUND")
    
    return True

def test_gui_creation():
    """Test GUI creation step by step"""
    print("🖥️ Testing GUI creation...")
    
    try:
        print("  Creating root window...")
        root = tk.Tk()
        print("  ✅ Root window created")
        
        print("  Setting window properties...")
        root.title("MINIMAL TEST - Fixacar SKU Predictor")
        root.geometry("800x600")
        print("  ✅ Window properties set")
        
        print("  Creating test widgets...")
        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        label = ttk.Label(frame, text="🎉 MINIMAL TEST SUCCESS!", font=("Arial", 16, "bold"))
        label.pack(pady=20)
        
        info_text = tk.Text(frame, height=15, width=80)
        info_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add test information
        test_info = """
MINIMAL TEST RESULTS:

✅ All imports working correctly
✅ File access working
✅ GUI creation successful

This means the core application should work!

The issue might be in:
1. Complex initialization code
2. Model loading
3. Database operations
4. Threading operations

Next steps:
1. If you see this window, the basic app works
2. We can gradually add more features
3. Identify exactly what causes the hang
"""
        
        info_text.insert(tk.END, test_info)
        info_text.config(state=tk.DISABLED)
        
        button = ttk.Button(frame, text="Close Test", command=root.quit)
        button.pack(pady=10)
        
        print("  ✅ Test widgets created")
        
        print("🚀 Starting GUI main loop...")
        print("=" * 50)
        print("GUI SHOULD NOW BE VISIBLE!")
        
        root.mainloop()
        
        print("✅ GUI closed normally")
        return True
        
    except Exception as e:
        print(f"❌ GUI creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 MINIMAL TEST VERSION")
    print("This version tests each component separately")
    print("=" * 50)
    
    try:
        # Test imports
        if not test_imports():
            print("❌ Import test failed")
            input("Press Enter to exit...")
            return False
        
        # Test custom imports
        if not test_custom_imports():
            print("❌ Custom import test failed")
            input("Press Enter to exit...")
            return False
        
        # Test file access
        test_file_access()
        
        # Test GUI
        if not test_gui_creation():
            print("❌ GUI test failed")
            input("Press Enter to exit...")
            return False
        
        print("✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
'''
    
    with open('src/minimal_test.py', 'w', encoding='utf-8') as f:
        f.write(minimal_content)
    
    print("✅ Minimal test file created: src/minimal_test.py")

def create_minimal_spec():
    """Create a minimal spec file for testing"""
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

# Add src directory to path for module discovery
src_path = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, src_path)

# Essential data collection only
numpy_datas = collect_data_files('numpy', include_py_files=True)
pandas_datas = collect_data_files('pandas', include_py_files=True)
torch_datas = collect_data_files('torch', include_py_files=True)

# Essential binary collection only
numpy_binaries = collect_dynamic_libs('numpy')
pandas_binaries = collect_dynamic_libs('pandas')
torch_binaries = collect_dynamic_libs('torch')

block_cipher = None

a = Analysis(
    ['src/minimal_test.py'],
    pathex=[
        os.path.join(os.getcwd(), 'src'),
        os.path.join(os.getcwd(), 'src', 'models'),
        os.path.join(os.getcwd(), 'src', 'utils'),
        os.getcwd()
    ],
    binaries=numpy_binaries + pandas_binaries + torch_binaries,
    datas=[
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/Text_Processing_Rules.xlsx', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/Maestro.xlsx', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/processed_consolidado.db', 'Source_Files'),
        ('src/models', 'src/models'),
        ('src/utils', 'src/utils'),
        ('src', 'src'),
    ] + numpy_datas + pandas_datas + torch_datas,
    hiddenimports=[
        # Core modules
        'tkinter', 'tkinter.ttk', 'tkinter.messagebox',
        
        # Essential scientific packages
        'numpy', 'numpy.core', 'numpy.core._multiarray_umath',
        'pandas', 'pandas._libs', 'pandas._libs.tslibs',
        'torch', 'torch.nn', 'torch.optim',
        'sklearn', 'openpyxl',
        
        # Custom modules
        'utils', 'utils.dummy_tokenizer',
        'models', 'models.sku_nn_pytorch',
        'src.utils', 'src.utils.dummy_tokenizer',
        'src.models', 'src.models.sku_nn_pytorch',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest', 'test'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Fixacar_MINIMAL_TEST',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open('Fixacar_MINIMAL_TEST.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("✅ Minimal spec file created")

def build_minimal_test():
    """Build the minimal test executable"""
    
    print("🔨 Building minimal test executable...")
    
    # Create minimal files
    create_minimal_main()
    create_minimal_spec()
    
    # Build the executable
    cmd = [
        'venv\\Scripts\\python.exe', '-m', 'PyInstaller',
        '--distpath', 'Fixacar_NUCLEAR_DEPLOYMENT\\Fixacar_SKU_Predictor_CLIENT',
        '--workpath', 'temp_build_minimal',
        '--clean',
        'Fixacar_MINIMAL_TEST.spec'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Minimal test build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Minimal test build failed: {e}")
        print("Error output:", e.stderr)
        return False

def create_minimal_batch():
    """Create a minimal test batch file"""
    
    batch_content = '''@echo off
echo.
echo ==========================================
echo   Fixacar SKU Predictor - MINIMAL TEST
echo ==========================================
echo.
echo This minimal test will help identify
echo exactly where the application is hanging.
echo.

cd /d "%~dp0"

if not exist "Fixacar_MINIMAL_TEST.exe" (
    echo ERROR: Fixacar_MINIMAL_TEST.exe not found!
    pause
    exit /b 1
)

echo Starting minimal test...
echo.

"Fixacar_MINIMAL_TEST.exe"

echo.
echo Test completed.
pause'''
    
    batch_path = "Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Fixacar_MINIMAL_TEST.bat"
    with open(batch_path, 'w', encoding='utf-8') as f:
        f.write(batch_content)
    
    print(f"✅ Minimal test batch file created: {batch_path}")

def main():
    """Main function"""
    
    print("🧪 Creating Minimal Test Version")
    print("=" * 50)
    
    if not os.path.exists('src'):
        os.makedirs('src')
    
    if build_minimal_test():
        create_minimal_batch()
        
        print("\n✅ Minimal test executable created successfully!")
        print("\n📋 Next steps:")
        print("1. Copy these files to the client laptop:")
        print("   - Fixacar_MINIMAL_TEST.exe")
        print("   - Fixacar_MINIMAL_TEST.bat")
        print("   - Source_Files folder")
        print("2. Run Fixacar_MINIMAL_TEST.bat on the client laptop")
        print("3. This will show exactly where the issue occurs")
        print("4. Report back what you see in the output")
        
        return True
    else:
        print("\n❌ Failed to create minimal test executable")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
