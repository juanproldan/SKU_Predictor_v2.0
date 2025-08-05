# -*- mode: python ; coding: utf-8 -*-

# =============================================================================
# Fixacar_Part_Finder_ROBUST.spec for Fixacar Application
# Solution Architect's Robust Approach
# =============================================================================

import os
from PyInstaller.utils.hooks import get_module_file_attribute

# --- STEP 1: DEFINE PATHS AND HIDDEN IMPORTS ---

# Full path to the spaCy model found in previous step
spacy_model_path = r'C:\Users\juanp\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\es_core_news_sm'

# BULLETPROOF hidden imports - every possible module that could be missing
hidden_imports = [
    # Core scipy array_api_compat modules (THE CRITICAL MISSING ONES)
    'scipy._lib.array_api_compat',
    'scipy._lib.array_api_compat.common',
    'scipy._lib.array_api_compat.numpy',
    'scipy._lib.array_api_compat.numpy.fft',
    'scipy._lib.array_api_compat.numpy.linalg',
    'scipy._lib.array_api_compat.numpy._aliases',
    'scipy._lib.array_api_compat.numpy._info',
    'scipy._lib.array_api_compat.numpy._typing',
    'scipy._lib.array_api_compat.torch',
    'scipy._lib.array_api_compat.cupy',
    'scipy._lib.array_api_compat.dask',
    'scipy._lib.array_api_compat._internal',

    # sklearn.externals array_api_compat modules (THE NEW CRITICAL MISSING ONES)
    'sklearn.externals.array_api_compat',
    'sklearn.externals.array_api_compat.common',
    'sklearn.externals.array_api_compat.numpy',
    'sklearn.externals.array_api_compat.numpy.fft',
    'sklearn.externals.array_api_compat.numpy.linalg',
    'sklearn.externals.array_api_compat.numpy._aliases',
    'sklearn.externals.array_api_compat.numpy._info',
    'sklearn.externals.array_api_compat.numpy._typing',
    'sklearn.externals.array_api_compat.torch',

    # Alternative array_api_compat paths
    'array_api_compat',
    'array_api_compat.common',
    'array_api_compat.numpy',
    'array_api_compat.numpy.fft',
    'array_api_compat.numpy.linalg',
    'array_api_compat.torch',

    # sklearn modules (INCLUDING THE MISSING CYTHON UTILITIES)
    'sklearn._cyutility',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree',
    'sklearn.tree._utils',
    'sklearn.externals.joblib',

    # scipy modules
    'scipy._lib.messagestream',
    'scipy.sparse',
    'scipy.sparse.csgraph',
    'scipy.linalg',
    'scipy.special',

    # spaCy modules
    'spacy',
    'spacy.lang.es',
    'spacy.pipeline',
    'spacy.tokens',
    'spacy.vocab',
    'spacy.strings',
    'spacy.lookups',
    'es_core_news_sm',
    'thinc',
    'thinc.api',
    'thinc.backends',
    'thinc.backends.numpy_ops',
    'srsly',
    'srsly.msgpack',
    'srsly.msgpack._packer',

    # PyTorch modules
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',

    # Core Python and data modules
    'joblib',
    'pandas',
    'numpy',
    'numpy.fft',
    'numpy.linalg',
    'sqlite3',
    'openpyxl',
    'openpyxl.workbook',
    'openpyxl.worksheet',

    # GUI modules
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',

    # Additional modules that might be needed
    'collections',
    'datetime',
    'json',
    're',
    'os',
    'sys',
    'pathlib'
]

# --- STEP 2: CONFIGURE THE ANALYSIS ---

a = Analysis(
    ['src/main_app.py'],  # Your main application script
    pathex=[],
    binaries=[],
    datas=[
        # Bundle the spaCy model data (CRITICAL FOR SPACY)
        (spacy_model_path, 'es_core_news_sm'),

        # Include Source_Files directory
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files', 'Source_Files'),

        # Include models directory
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/models', 'models'),

        # Include scipy array_api_compat data files (CRITICAL FOR ARRAY_API_COMPAT)
        (r'C:\Users\juanp\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\scipy\_lib\array_api_compat', 'scipy/_lib/array_api_compat'),

        # Include sklearn.externals array_api_compat data files (CRITICAL FOR SKLEARN)
        (r'C:\Users\juanp\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\sklearn\externals\array_api_compat', 'sklearn/externals/array_api_compat'),

        # Include additional scipy data that might be needed
        (r'C:\Users\juanp\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\scipy\_lib', 'scipy/_lib'),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# --- STEP 3: CONFIGURE THE EXECUTABLE ---

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Fixacar_Part_Finder',  # Name of your final .exe file
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
