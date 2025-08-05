# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# Collect all critical dependencies
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all('pandas')
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
openpyxl_datas, openpyxl_binaries, openpyxl_hiddenimports = collect_all('openpyxl')
requests_datas, requests_binaries, requests_hiddenimports = collect_all('requests')

a = Analysis(
    ['src/main_app.py'],
    pathex=['src'],
    binaries=numpy_binaries + pandas_binaries + sklearn_binaries + torch_binaries + openpyxl_binaries + requests_binaries,
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
    ] + numpy_datas + pandas_datas + sklearn_datas + torch_datas + openpyxl_datas + requests_datas,
    hiddenimports=numpy_hiddenimports + pandas_hiddenimports + sklearn_hiddenimports + torch_hiddenimports + openpyxl_hiddenimports + requests_hiddenimports + [
        # Force include ALL numpy submodules
        'numpy', 'numpy.core', 'numpy.core._multiarray_umath', 'numpy.core.multiarray',
        'numpy.core.umath', 'numpy._distributor_init', 'numpy.linalg', 'numpy.fft',
        'numpy.random', 'numpy.random._pickle', 'numpy.lib', 'numpy.lib.format',
        'numpy.ma', 'numpy.polynomial',

        # Force include ALL pandas submodules
        'pandas', 'pandas._libs', 'pandas._libs.lib', 'pandas._libs.hashtable',
        'pandas._libs.tslib', 'pandas._libs.tslibs', 'pandas._libs.interval',
        'pandas.core', 'pandas.io', 'pandas.io.excel', 'pandas.io.common',

        # Force include ALL sklearn submodules
        'sklearn', 'sklearn.utils', 'sklearn.utils._cython_blas', 'sklearn.tree._utils',
        'sklearn.ensemble._forest', 'sklearn.tree', 'sklearn.tree._tree',
        'sklearn.linear_model', 'sklearn.ensemble', 'sklearn.preprocessing',

        # Force include ALL torch submodules
        'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils',
        'torch._C', 'torch.jit', 'torch.autograd', 'torch.serialization',
        'torch.nn.modules', 'torch.nn.modules.linear', 'torch.nn.modules.activation',

        # Project-specific modules (CRITICAL for imports)
        'src.models.sku_nn_pytorch', 'src.utils.text_utils', 'src.utils.dummy_tokenizer',
        'src.utils.pytorch_tokenizer', 'src.utils.fuzzy_matcher', 'src.utils.logging_config',
        'src.utils.optimized_database', 'src.utils.optimized_startup', 'src.utils.spacy_text_processor',
        'src.utils.year_range_database', 'src.train_vin_predictor', 'src.unified_consolidado_processor',

        # System and utility libraries
        'joblib', 'sqlite3', 'json', 'pickle', 'threading', 'logging', 'datetime',
        'collections', 'itertools', 'functools', 'operator', 'math', 'statistics',

        # GUI libraries
        'tkinter', 'tkinter.ttk', 'tkinter.messagebox', 'tkinter.filedialog', 'tkinter.font',

        # Network and data libraries
        'requests', 'urllib3', 'urllib', 'urllib.request', 'urllib.parse', 'http.client',

        # Additional critical imports for standalone execution
        'pkg_resources', 'setuptools', 'distutils', 'importlib', 'importlib.util'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest'],
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
    name='Fixacar_SKU_Predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
