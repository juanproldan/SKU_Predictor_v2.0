# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/unified_consolidado_processor.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('Fixacar_SKU_Predictor_CLIENT/Source_Files', 'Source_Files'),
        ('Fixacar_SKU_Predictor_CLIENT/logs', 'logs'),
        ('src/utils', 'src/utils'),
        ('src/core', 'src/core'),
    ],
    hiddenimports=[
        # Pandas - CRITICAL dependencies
        'pandas', 'pandas._libs', 'pandas._libs.tslibs', 'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.np_datetime', 'pandas._libs.tslibs.nattype', 'pandas._libs.skiplist',
        'pandas.core', 'pandas.io', 'pandas.io.excel', 'pandas.io.common',

        # NumPy - CRITICAL for all numerical operations
        'numpy', 'numpy.core', 'numpy.core._multiarray_umath', 'numpy.core._multiarray_tests',
        'numpy.linalg', 'numpy.fft', 'numpy.random', 'numpy.random._pickle', 'numpy.lib',
        'numpy.lib.format', 'numpy.ma', 'numpy.polynomial',

        # Excel/Office support
        'openpyxl', 'openpyxl.cell', 'openpyxl.cell._writer', 'openpyxl.workbook',
        'openpyxl.worksheet', 'openpyxl.styles', 'openpyxl.utils',

        # Database support
        'sqlite3', 'sqlite3.dbapi2',

        # Text processing and NLP
        'spacy', 'spacy.lang', 'spacy.lang.es', 'spacy.tokens', 'spacy.matcher',
        're', 'unicodedata', 'difflib',

        # Project-specific modules
        'src.utils.text_utils', 'src.utils.spacy_text_processor', 'src.utils.logging_config',
        'src.utils.optimized_database', 'src.core.vin_validator',

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
    name='2. Fixacar_Data_Processor',
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
