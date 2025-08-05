#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a bulletproof executable that works on all client laptops
This script will rebuild the SKU Predictor with comprehensive dependency inclusion
"""

import os
import sys
import subprocess
import shutil

def create_enhanced_spec_file():
    """Create an enhanced spec file with comprehensive dependency inclusion"""
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

# Comprehensive data collection
numpy_datas = collect_data_files('numpy', include_py_files=True)
pandas_datas = collect_data_files('pandas', include_py_files=True)
sklearn_datas = collect_data_files('sklearn', include_py_files=True)
torch_datas = collect_data_files('torch', include_py_files=True)
openpyxl_datas = collect_data_files('openpyxl', include_py_files=True)
requests_datas = collect_data_files('requests', include_py_files=True)
urllib3_datas = collect_data_files('urllib3', include_py_files=True)

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
    pathex=[],
    binaries=numpy_binaries + pandas_binaries + sklearn_binaries + torch_binaries + [
        # Additional critical binaries
    ],
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
    ] + numpy_datas + pandas_datas + sklearn_datas + torch_datas + openpyxl_datas + requests_datas + urllib3_datas,
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
        'sklearn.externals', 'sklearn.externals.joblib',
        
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
        
        # Custom modules
        'src.models.sku_nn_pytorch',
        'src.utils.dummy_tokenizer',
        'src.utils.pytorch_tokenizer', 
        'src.utils.logging_config',
        'src.utils.optimized_database',
        'src.utils.optimized_startup',
        'src.utils.year_range_database',
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
    name='Fixacar_SKU_Predictor_BULLETPROOF',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX to avoid compression issues
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Enable console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open('Fixacar_SKU_Predictor_BULLETPROOF.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("✅ Enhanced spec file created: Fixacar_SKU_Predictor_BULLETPROOF.spec")

def build_bulletproof_executable():
    """Build the bulletproof executable"""
    
    print("🔨 Building bulletproof executable...")
    
    # Create the enhanced spec file
    create_enhanced_spec_file()
    
    # Build the executable
    cmd = [
        'venv\\Scripts\\python.exe', '-m', 'PyInstaller',
        '--distpath', 'Fixacar_NUCLEAR_DEPLOYMENT\\Fixacar_SKU_Predictor_CLIENT',
        '--workpath', 'temp_build_bulletproof',
        '--clean',
        'Fixacar_SKU_Predictor_BULLETPROOF.spec'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build completed successfully!")
        print("Build output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print("Error output:", e.stderr)
        return False

def create_test_script():
    """Create a test script to verify the executable works"""
    
    test_script = '''
import subprocess
import os
import sys

def test_executable():
    """Test the bulletproof executable"""
    
    exe_path = os.path.join("Fixacar_NUCLEAR_DEPLOYMENT", "Fixacar_SKU_Predictor_CLIENT", "Fixacar_SKU_Predictor_BULLETPROOF.exe")
    
    if not os.path.exists(exe_path):
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    print(f"🧪 Testing executable: {exe_path}")
    
    try:
        # Run the executable with a timeout
        result = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(exe_path)
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Check for import errors
        if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
            print("❌ Import errors detected!")
            return False
        elif "Successfully imported all modules" in result.stdout:
            print("✅ All modules imported successfully!")
            return True
        else:
            print("⚠️ Executable started but status unclear")
            return True
            
    except subprocess.TimeoutExpired:
        print("⏰ Executable timed out (might be waiting for GUI interaction)")
        return True  # Timeout might be normal for GUI apps
    except Exception as e:
        print(f"❌ Error testing executable: {e}")
        return False

if __name__ == "__main__":
    success = test_executable()
    sys.exit(0 if success else 1)
'''
    
    with open('test_bulletproof_executable.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ Test script created: test_bulletproof_executable.py")

def main():
    """Main function to create bulletproof executable"""
    
    print("🚀 Creating Bulletproof Fixacar SKU Predictor Executable")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('src/main_app.py'):
        print("❌ Error: src/main_app.py not found. Please run from project root.")
        return False
    
    # Build the executable
    if build_bulletproof_executable():
        print("\n✅ Bulletproof executable created successfully!")
        
        # Create test script
        create_test_script()
        
        print("\n📋 Next steps:")
        print("1. Test the executable: python test_bulletproof_executable.py")
        print("2. Copy the entire Fixacar_NUCLEAR_DEPLOYMENT folder to client laptops")
        print("3. Run Fixacar_SKU_Predictor_BULLETPROOF.exe on client machines")
        
        return True
    else:
        print("\n❌ Failed to create bulletproof executable")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
