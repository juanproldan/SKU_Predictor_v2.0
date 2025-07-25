# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Get the current directory (where the spec file is located)
current_dir = os.path.dirname(os.path.abspath(SPEC))

# Define paths
src_path = os.path.join(current_dir, 'src')
source_files_path = os.path.join(current_dir, 'Source_Files')

# Add src to Python path
sys.path.insert(0, src_path)

# Define data files to include
datas = [
    (source_files_path, 'Source_Files'),
    (src_path, 'src'),
]

# Define hidden imports
hiddenimports = [
    'sklearn',
    'sklearn.ensemble',
    'sklearn.linear_model', 
    'sklearn.model_selection',
    'sklearn.preprocessing',
    'sklearn.metrics',
    'sklearn.feature_extraction',
    'sklearn.feature_extraction.text',
    'pandas',
    'numpy',
    'openpyxl',
    'sqlite3',
    'joblib',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'logging',
    'datetime',
    'json',
    'os',
    'sys',
    'pathlib',
    're',
    'unicodedata',
    'difflib',
    'collections',
    'tqdm',
    'pickle',
]

a = Analysis(
    ['src/train_sku_nn_predictor_pytorch_optimized.py'],
    pathex=[current_dir, src_path],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Fixacar_SKU_Trainer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
