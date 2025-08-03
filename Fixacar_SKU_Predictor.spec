# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/main_app.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('Fixacar_SKU_Predictor_CLIENT/Source_Files/Text_Processing_Rules.xlsx', 'Source_Files'),
        ('Fixacar_SKU_Predictor_CLIENT/Source_Files/Maestro.xlsx', 'Source_Files'),
        ('Fixacar_SKU_Predictor_CLIENT/Source_Files/processed_consolidado.db', 'Source_Files'),
        ('Fixacar_SKU_Predictor_CLIENT/Source_Files/Consolidado.json', 'Source_Files'),
        ('Fixacar_SKU_Predictor_CLIENT/models', 'models'),
        ('src/models', 'src/models'),
        ('src/utils', 'src/utils'),
        ('src/core', 'src/core'),
        ('src/gui', 'src/gui'),
    ],
    hiddenimports=[
        # Core ML/Data libraries
        'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.tree._utils',
        'sklearn.ensemble._forest', 'sklearn.tree', 'sklearn.tree._tree',
        'pandas._libs.tslibs.timedeltas', 'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.nattype', 'pandas._libs.skiplist',
        'openpyxl.cell._writer', 'openpyxl.workbook', 'openpyxl.worksheet',

        # PyTorch and NumPy (CRITICAL for standalone)
        'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils',
        'torch._C', 'torch._C._nn', 'torch.jit', 'torch.autograd',
        'numpy', 'numpy.core', 'numpy.core._multiarray_umath', 'numpy.core._multiarray_tests',
        'numpy.linalg', 'numpy.fft', 'numpy.random', 'numpy.random._pickle',

        # Project-specific modules (CRITICAL for imports)
        'src.models.sku_nn_pytorch', 'src.utils.text_utils', 'src.utils.dummy_tokenizer', 'src.utils.pytorch_tokenizer',
        'src.utils.fuzzy_matcher', 'src.utils.logging_config', 'src.utils.optimized_database', 'src.utils.optimized_startup',
        'src.utils.spacy_text_processor', 'src.utils.year_range_database',
        'src.train_vin_predictor', 'src.unified_consolidado_processor',

        # Other essentials
        'joblib', 'sqlite3', 'tkinter', 'tkinter.ttk', 'tkinter.messagebox',
        'tkinter.filedialog', 'requests', 'urllib3', 'json', 'pickle', 'threading'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
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
