#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a FIXED executable that properly includes all custom modules
The issue was that PyInstaller wasn't finding the custom modules in src/
"""

import os
import sys
import subprocess

def create_fixed_spec():
    """Create a fixed spec file that properly includes custom modules"""
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

# Add src directory to path for module discovery
src_path = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, src_path)

# Comprehensive data collection
numpy_datas = collect_data_files('numpy', include_py_files=True)
pandas_datas = collect_data_files('pandas', include_py_files=True)
sklearn_datas = collect_data_files('sklearn', include_py_files=True)
torch_datas = collect_data_files('torch', include_py_files=True)
openpyxl_datas = collect_data_files('openpyxl', include_py_files=True)

# Comprehensive binary collection
numpy_binaries = collect_dynamic_libs('numpy')
pandas_binaries = collect_dynamic_libs('pandas')
sklearn_binaries = collect_dynamic_libs('sklearn')
torch_binaries = collect_dynamic_libs('torch')

# Comprehensive hidden imports
numpy_hidden = collect_submodules('numpy')
pandas_hidden = collect_submodules('pandas')
sklearn_hidden = collect_submodules('sklearn')
torch_hidden = collect_submodules('torch')

block_cipher = None

a = Analysis(
    ['src/main_app.py'],
    pathex=[
        os.path.join(os.getcwd(), 'src'),
        os.path.join(os.getcwd(), 'src', 'models'),
        os.path.join(os.getcwd(), 'src', 'utils'),
        os.path.join(os.getcwd(), 'src', 'core'),
        os.path.join(os.getcwd(), 'src', 'gui'),
        os.getcwd()
    ],
    binaries=numpy_binaries + pandas_binaries + sklearn_binaries + torch_binaries,
    datas=[
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/Text_Processing_Rules.xlsx', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/Maestro.xlsx', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/processed_consolidado.db', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/Consolidado.json', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/models', 'models'),
        ('src/models', 'src/models'),
        ('src/utils', 'src/utils'),
        ('src/core', 'src/core'),
        ('src/gui', 'src/gui'),
        ('src', 'src'),  # Include entire src directory
    ] + numpy_datas + pandas_datas + sklearn_datas + torch_datas + openpyxl_datas,
    hiddenimports=[
        # Core Python modules
        'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.font',
        
        # NumPy comprehensive
        'numpy', 'numpy.core', 'numpy.core._multiarray_umath', 'numpy.core._multiarray_tests',
        'numpy.core._dtype_ctypes', 'numpy.linalg', 'numpy.linalg.lapack_lite',
        'numpy.random', 'numpy.random.mtrand', 'numpy.fft', 'numpy.polynomial',
        'numpy._core', 'numpy._core._dtype_ctypes', 'numpy._core._multiarray_tests',
        'numpy._core._exceptions', 'numpy._core._multiarray_umath',
        
        # Pandas comprehensive  
        'pandas', 'pandas._libs', 'pandas._libs.tslibs', 'pandas._libs.tslibs.base',
        'pandas.core', 'pandas.core.arrays', 'pandas.io', 'pandas.io.formats',
        'pandas.io.formats.style', 'pandas.plotting', 'pandas.io.clipboard',
        'pandas._libs.algos', 'pandas._libs.groupby', 'pandas._libs.hashing',
        'pandas._libs.hashtable', 'pandas._libs.index', 'pandas._libs.internals',
        'pandas._libs.join', 'pandas._libs.lib', 'pandas._libs.missing',
        'pandas._libs.parsers', 'pandas._libs.reduction', 'pandas._libs.reshape',
        'pandas._libs.sparse', 'pandas._libs.testing', 'pandas._libs.window',
        
        # Scikit-learn comprehensive
        'sklearn', 'sklearn.utils', 'sklearn.utils._cython_blas',
        'sklearn.metrics', 'sklearn.metrics.cluster', 'sklearn.metrics.pairwise',
        'sklearn.neighbors', 'sklearn.linear_model', 'sklearn.cluster',
        
        # PyTorch comprehensive
        'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils',
        'torch.utils.data', 'torch.autograd', 'torch.cuda', 'torch.jit',
        'torch._C', 'torch._utils', 'torch.multiprocessing',
        
        # Other critical modules
        'openpyxl', 'openpyxl.utils', 'openpyxl.utils.dataframe', 'openpyxl.utils.inference',
        'requests', 'requests.help', 'urllib3', 'certifi',
        'sqlite3', 'json', 'pickle', 'joblib', 'threading', 'multiprocessing',
        'logging', 'logging.handlers', 'datetime', 'collections', 'itertools',
        'functools', 'operator', 'math', 'statistics', 're', 'string',
        'pathlib', 'glob', 'shutil', 'tempfile', 'zipfile', 'gzip',
        
        # CRITICAL: Custom modules - explicit imports
        'models',
        'models.sku_nn_pytorch',
        'utils',
        'utils.dummy_tokenizer',
        'utils.pytorch_tokenizer', 
        'utils.logging_config',
        'utils.optimized_database',
        'utils.optimized_startup',
        'utils.year_range_database',
        'utils.text_utils',
        'train_vin_predictor',
        'unified_consolidado_processor',
        
        # Also try with src prefix
        'src',
        'src.models',
        'src.models.sku_nn_pytorch',
        'src.utils',
        'src.utils.dummy_tokenizer',
        'src.utils.pytorch_tokenizer', 
        'src.utils.logging_config',
        'src.utils.optimized_database',
        'src.utils.optimized_startup',
        'src.utils.year_range_database',
        'src.utils.text_utils',
        'src.train_vin_predictor',
        'src.unified_consolidado_processor',
        
    ] + numpy_hidden + pandas_hidden + sklearn_hidden + torch_hidden,
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
    name='Fixacar_SKU_Predictor_FIXED',
    debug=True,  # Keep debug mode
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console visible
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open('Fixacar_SKU_Predictor_FIXED.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("✅ Fixed spec file created")

def build_fixed_executable():
    """Build the fixed executable"""
    
    print("🔨 Building FIXED executable...")
    
    # Create the fixed spec file
    create_fixed_spec()
    
    # Build the executable
    cmd = [
        'venv\\Scripts\\python.exe', '-m', 'PyInstaller',
        '--distpath', 'Fixacar_NUCLEAR_DEPLOYMENT\\Fixacar_SKU_Predictor_CLIENT',
        '--workpath', 'temp_build_fixed',
        '--clean',
        'Fixacar_SKU_Predictor_FIXED.spec'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Fixed build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fixed build failed: {e}")
        print("Error output:", e.stderr)
        return False

def create_fixed_batch():
    """Create a fixed batch file"""
    
    batch_content = '''@echo off
echo.
echo ==========================================
echo   Fixacar SKU Predictor - FIXED VERSION
echo ==========================================
echo.
echo This FIXED version properly includes all
echo custom modules and should work correctly.
echo.

REM Change to the directory where the batch file is located
cd /d "%~dp0"

REM Check if the executable exists
if not exist "Fixacar_SKU_Predictor_FIXED.exe" (
    echo ERROR: Fixacar_SKU_Predictor_FIXED.exe not found!
    pause
    exit /b 1
)

echo Starting FIXED version...
echo.

REM Run the fixed executable
"Fixacar_SKU_Predictor_FIXED.exe"

echo.
echo Application closed.
pause'''
    
    batch_path = "Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Fixacar_SKU_Predictor_FIXED.bat"
    with open(batch_path, 'w', encoding='utf-8') as f:
        f.write(batch_content)
    
    print(f"✅ Fixed batch file created: {batch_path}")

def main():
    """Main function to create fixed executable"""
    
    print("🛠️ Creating FIXED Fixacar SKU Predictor")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('src/main_app.py'):
        print("❌ Error: src/main_app.py not found. Please run from project root.")
        return False
    
    # Build the fixed executable
    if build_fixed_executable():
        print("\n✅ FIXED executable created successfully!")
        
        # Create fixed batch file
        create_fixed_batch()
        
        print("\n📋 Next steps:")
        print("1. Test locally: Fixacar_NUCLEAR_DEPLOYMENT\\Fixacar_SKU_Predictor_CLIENT\\Fixacar_SKU_Predictor_FIXED.exe")
        print("2. Copy the FIXED files to the client laptop:")
        print("   - Fixacar_SKU_Predictor_FIXED.exe")
        print("   - Fixacar_SKU_Predictor_FIXED.bat")
        print("3. Run the FIXED batch file on the client laptop")
        print("4. This version should properly load all custom modules")
        
        return True
    else:
        print("\n❌ Failed to create FIXED executable")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
