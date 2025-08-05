# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# Collect all numpy, pandas, sklearn data and binaries
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all('pandas')
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')
openpyxl_datas, openpyxl_binaries, openpyxl_hiddenimports = collect_all('openpyxl')

a = Analysis(
    ['src/unified_consolidado_processor.py'],
    pathex=['src'],
    binaries=numpy_binaries + pandas_binaries + sklearn_binaries + openpyxl_binaries,
    datas=[
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/logs', 'logs'),
        ('src/utils', 'src/utils'),
        ('src/core', 'src/core'),
    ] + numpy_datas + pandas_datas + sklearn_datas + openpyxl_datas,
    hiddenimports=numpy_hiddenimports + pandas_hiddenimports + sklearn_hiddenimports + openpyxl_hiddenimports + [
        # Force include ALL numpy submodules
        'numpy', 'numpy.core', 'numpy.core._multiarray_umath', 'numpy.core._multiarray_tests',
        'numpy.linalg', 'numpy.fft', 'numpy.random', 'numpy.random._pickle', 'numpy.lib',
        'numpy.lib.format', 'numpy.ma', 'numpy.polynomial', 'numpy.core.multiarray',
        'numpy.core.umath', 'numpy._distributor_init', 'numpy.core._dtype_ctypes',
        'numpy.core._internal', 'numpy.core._methods', 'numpy.core.arrayprint',
        'numpy.core.defchararray', 'numpy.core.einsumfunc', 'numpy.core.fromnumeric',
        'numpy.core.function_base', 'numpy.core.getlimits', 'numpy.core.machar',
        'numpy.core.memmap', 'numpy.core.numeric', 'numpy.core.numerictypes',
        'numpy.core.overrides', 'numpy.core.records', 'numpy.core.shape_base',

        # Force include ALL pandas submodules
        'pandas', 'pandas._libs', 'pandas._libs.tslibs', 'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.np_datetime', 'pandas._libs.tslibs.nattype', 'pandas._libs.skiplist',
        'pandas.core', 'pandas.io', 'pandas.io.excel', 'pandas.io.common', 'pandas._libs.lib',
        'pandas._libs.hashtable', 'pandas._libs.tslib', 'pandas._libs.interval',
        'pandas._libs.join', 'pandas._libs.indexing', 'pandas._libs.algos',
        'pandas._libs.groupby', 'pandas._libs.reshape', 'pandas._libs.sparse',
        'pandas._libs.ops', 'pandas._libs.parsers', 'pandas._libs.writers',

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
