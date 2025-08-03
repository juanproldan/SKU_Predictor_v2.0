# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/train_vin_predictor.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('Fixacar_SKU_Predictor_CLIENT/Source_Files', 'Source_Files'),
        ('Fixacar_SKU_Predictor_CLIENT/models', 'models'),
        ('Fixacar_SKU_Predictor_CLIENT/logs', 'logs'),
        ('src/utils', 'src/utils'),
        ('src/core', 'src/core'),
    ],
    hiddenimports=[
        # Core ML/Data libraries - CRITICAL for model training
        'sklearn', 'sklearn.utils', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs',
        'sklearn.tree._utils', 'sklearn.ensemble._forest', 'sklearn.tree', 'sklearn.tree._tree',
        'sklearn.linear_model', 'sklearn.ensemble', 'sklearn.preprocessing',
        'sklearn.model_selection', 'sklearn.metrics', 'sklearn.pipeline',

        # Pandas - CRITICAL dependencies
        'pandas', 'pandas._libs', 'pandas._libs.tslibs', 'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.np_datetime', 'pandas._libs.tslibs.nattype', 'pandas._libs.skiplist',
        'pandas.core', 'pandas.io', 'pandas.io.excel', 'pandas.io.common',

        # NumPy - CRITICAL for all numerical operations
        'numpy', 'numpy.core', 'numpy.core._multiarray_umath', 'numpy.core._multiarray_tests',
        'numpy.linalg', 'numpy.fft', 'numpy.random', 'numpy.random._pickle', 'numpy.lib',
        'numpy.lib.format', 'numpy.ma', 'numpy.polynomial',

        # PyTorch - CRITICAL for neural networks
        'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils',
        'torch._C', 'torch._C._nn', 'torch.jit', 'torch.autograd', 'torch.serialization',
        'torch.nn.modules', 'torch.nn.modules.linear', 'torch.nn.modules.activation',

        # Model persistence
        'joblib', 'pickle', 'dill',

        # Database support
        'sqlite3', 'sqlite3.dbapi2',

        # Project-specific modules
        'src.utils.logging_config', 'src.utils.optimized_database', 'src.core.vin_validator',
        'src.unified_consolidado_processor',

        # System and utility libraries
        'json', 'logging', 'datetime', 'os', 'sys', 'pathlib', 'argparse',
        'collections', 'itertools', 'functools', 'operator', 'math', 'statistics',
        'threading', 'multiprocessing', 'concurrent.futures',

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
    name='3. Fixacar_VIN_Trainer',
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
