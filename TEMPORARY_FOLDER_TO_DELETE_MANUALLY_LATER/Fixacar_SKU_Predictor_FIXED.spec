# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

# Add src directory to path for module discovery
src_path = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, src_path)

# Comprehensive data collection
numpy_datas = collect_data_files('numpy', include_py_files=True)
pandas_datas = collect_data_files('pandas', include_py_files=True)
sklearn_datas = collect_data_files('sklearn', include_py_files=True)
torch_datas = collect_data_files('torch', include_py_files=True)
openpyxl_datas = collect_data_files('openpyxl', include_py_files=True)
# spaCy data collection with explicit paths
try:
    spacy_datas = collect_data_files('spacy', include_py_files=True)
    print("✅ spaCy data files collected")
except Exception as e:
    print(f"⚠️ spaCy data collection failed: {e}")
    spacy_datas = []

try:
    es_model_datas = collect_data_files('es_core_news_sm', include_py_files=True)
    print("✅ Spanish model data files collected")
except Exception as e:
    print(f"⚠️ Spanish model data collection failed: {e}")
    es_model_datas = []

# Additional spaCy data paths
import spacy
import es_core_news_sm
spacy_path = os.path.dirname(spacy.__file__)
es_model_path = os.path.dirname(es_core_news_sm.__file__)

additional_spacy_datas = [
    (spacy_path, 'spacy'),
    (es_model_path, 'es_core_news_sm'),
]

# Comprehensive binary collection
numpy_binaries = collect_dynamic_libs('numpy')
pandas_binaries = collect_dynamic_libs('pandas')
sklearn_binaries = collect_dynamic_libs('sklearn')
torch_binaries = collect_dynamic_libs('torch')
spacy_binaries = collect_dynamic_libs('spacy')

# Comprehensive hidden imports
numpy_hidden = collect_submodules('numpy')
pandas_hidden = collect_submodules('pandas')
sklearn_hidden = collect_submodules('sklearn')
torch_hidden = collect_submodules('torch')
spacy_hidden = collect_submodules('spacy')
es_model_hidden = collect_submodules('es_core_news_sm')

block_cipher = None

a = Analysis(
    ['src/main_app.py'],
    pathex=[
        os.path.join(os.getcwd(), 'src'),
        os.path.join(os.getcwd(), 'src', 'models'),
        os.path.join(os.getcwd(), 'src', 'utils'),
        os.path.join(os.getcwd(), 'src', 'core'),
        os.path.join(os.getcwd(), 'src', 'gui'),
        os.getcwd()
    ],
    binaries=numpy_binaries + pandas_binaries + sklearn_binaries + torch_binaries + spacy_binaries,
    datas=[
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/Text_Processing_Rules.xlsx', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/Maestro.xlsx', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/processed_consolidado.db', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files/Consolidado.json', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/models', 'models'),
        ('src/models', 'src/models'),
        ('src/utils', 'src/utils'),
        ('src/utils/performance_improvements', 'src/utils/performance_improvements'),
        ('src/core', 'src/core'),
        ('src/gui', 'src/gui'),
        ('src', 'src'),  # Include entire src directory
    ] + numpy_datas + pandas_datas + sklearn_datas + torch_datas + openpyxl_datas + spacy_datas + es_model_datas + additional_spacy_datas,
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
        
        # PyTorch comprehensive
        'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils',
        'torch.utils.data', 'torch.autograd', 'torch.cuda', 'torch.jit',
        'torch._C', 'torch._utils', 'torch.multiprocessing',
        
        # spaCy comprehensive
        'spacy', 'spacy.lang', 'spacy.lang.es', 'spacy.lang.es.stop_words',
        'spacy.pipeline', 'spacy.tokens', 'spacy.vocab', 'spacy.strings',
        'spacy.matcher', 'spacy.util', 'spacy.cli', 'spacy.about',
        'es_core_news_sm', 'spacy.lang.es.lemmatizer',

        # Other critical modules
        'openpyxl', 'openpyxl.utils', 'openpyxl.utils.dataframe', 'openpyxl.utils.inference',
        'requests', 'requests.help', 'urllib3', 'certifi',
        'sqlite3', 'json', 'pickle', 'joblib', 'threading', 'multiprocessing',
        'logging', 'logging.handlers', 'datetime', 'collections', 'itertools',
        'functools', 'operator', 'math', 'statistics', 're', 'string',
        'pathlib', 'glob', 'shutil', 'tempfile', 'zipfile', 'gzip',
        
        # CRITICAL: Custom modules - explicit imports
        'models',
        'models.sku_nn_pytorch',
        'utils',
        'utils.dummy_tokenizer',
        'utils.pytorch_tokenizer',
        'utils.logging_config',
        'utils.optimized_database',
        'utils.optimized_startup',
        'utils.year_range_database',
        'utils.text_utils',
        'train_vin_predictor',
        'unified_consolidado_processor',

        # Performance improvements modules
        'utils.performance_improvements',
        'utils.performance_improvements.optimizations',
        'utils.performance_improvements.optimizations.database_optimizer',
        'utils.performance_improvements.optimizations.parallel_predictor',
        'utils.performance_improvements.enhanced_text_processing',
        'utils.performance_improvements.enhanced_text_processing.smart_text_processor',
        'utils.performance_improvements.enhanced_text_processing.equivalencias_analyzer',
        
        # Also try with src prefix
        'src',
        'src.models',
        'src.models.sku_nn_pytorch',
        'src.utils',
        'src.utils.dummy_tokenizer',
        'src.utils.pytorch_tokenizer',
        'src.utils.logging_config',
        'src.utils.optimized_database',
        'src.utils.optimized_startup',
        'src.utils.year_range_database',
        'src.utils.text_utils',
        'src.train_vin_predictor',
        'src.unified_consolidado_processor',

        # Performance improvements with src prefix
        'src.utils.performance_improvements',
        'src.utils.performance_improvements.optimizations',
        'src.utils.performance_improvements.optimizations.database_optimizer',
        'src.utils.performance_improvements.optimizations.parallel_predictor',
        'src.utils.performance_improvements.enhanced_text_processing',
        'src.utils.performance_improvements.enhanced_text_processing.smart_text_processor',
        'src.utils.performance_improvements.enhanced_text_processing.equivalencias_analyzer',
        
    ] + numpy_hidden + pandas_hidden + sklearn_hidden + torch_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest', 'test'],
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
    name='Fixacar_SKU_Predictor_FIXED',
    debug=True,  # Keep debug mode
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console visible
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
