# -*- mode: python ; coding: utf-8 -*-

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
