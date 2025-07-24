# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec file for Fixacar SKU Predictor
This spec file ensures all dependencies are properly included
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(SPEC))

# Define paths
src_path = os.path.join(current_dir, 'src')
data_path = os.path.join(current_dir, 'data')
models_path = os.path.join(current_dir, 'models')
source_files_path = os.path.join(current_dir, 'Source_Files')

# Collect all submodules for problematic packages
torch_submodules = collect_submodules('torch')
sklearn_submodules = collect_submodules('sklearn')
scipy_submodules = collect_submodules('scipy')
numpy_submodules = collect_submodules('numpy')
pandas_submodules = collect_submodules('pandas')

# Collect data files for packages that need them
torch_data = collect_data_files('torch')
sklearn_data = collect_data_files('sklearn')
scipy_data = collect_data_files('scipy')

# Define hidden imports - comprehensive list
hidden_imports = [
    # Core Python libraries
    'collections',
    'datetime',
    'json',
    're',
    'sqlite3',
    'time',
    'os',
    'sys',
    
    # Data processing
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.lib.format',
    'pandas',
    'pandas._libs.tslibs.base',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.timestamps',
    'openpyxl',
    'openpyxl.workbook',
    'openpyxl.worksheet',
    'joblib',
    
    # Machine Learning - Scipy
    'scipy',
    'scipy.sparse',
    'scipy.sparse._base',
    'scipy.sparse._sputils',
    'scipy._lib',
    'scipy._lib._util',
    'scipy._lib._array_api',
    'scipy._lib.array_api_compat',
    'scipy._lib.array_api_compat.numpy',
    'scipy._lib.array_api_compat.numpy.fft',
    'scipy.special',
    'scipy.linalg',
    'scipy.optimize',

    # Machine Learning - Sklearn
    'sklearn',
    'sklearn.preprocessing',
    'sklearn.model_selection',
    'sklearn.metrics',
    'sklearn.naive_bayes',
    'sklearn.utils',
    'sklearn.utils._param_validation',
    'sklearn.utils.fixes',
    
    # PyTorch
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torch.optim.lr_scheduler',
    'torch.cuda',
    'torch.jit',
    'torch.serialization',
    
    # Text processing
    'fuzzywuzzy',
    'Levenshtein',
    
    # GUI
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.font',
    
    # Custom modules
    'models.sku_nn_pytorch',
    'utils.text_utils',
    'utils.dummy_tokenizer',
    'utils.pytorch_tokenizer',
]

# Add collected submodules
hidden_imports.extend(torch_submodules)
hidden_imports.extend(sklearn_submodules)
hidden_imports.extend(scipy_submodules)
hidden_imports.extend(numpy_submodules)
hidden_imports.extend(pandas_submodules)

# Remove duplicates
hidden_imports = list(set(hidden_imports))

# Define data files to include
datas = [
    (source_files_path, 'Source_Files'),
    (data_path, 'data'),
    (models_path, 'models'),
    (src_path, 'src'),
]

# Add collected data files
datas.extend(torch_data)
datas.extend(sklearn_data)
datas.extend(scipy_data)

# Analysis configuration
a = Analysis(
    ['src/main_app.py'],
    pathex=[current_dir, src_path],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[os.path.join(current_dir, 'hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'setuptools',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create the executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Fixacar_SKU_Predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
