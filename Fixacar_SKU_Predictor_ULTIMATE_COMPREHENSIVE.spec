# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import glob
from pathlib import Path

# Add the src directory to Python path
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# COMPREHENSIVE spaCy data collection with FIXED model paths
def collect_spacy_data():
    """Collect all spaCy data files and directories with proper model registration"""
    try:
        import spacy
        import es_core_news_sm

        spacy_dir = os.path.dirname(spacy.__file__)
        model_dir = os.path.dirname(es_core_news_sm.__file__)

        print(f"✅ spaCy directory: {spacy_dir}")
        print(f"✅ Spanish model directory: {model_dir}")

        # Collect all spaCy data files
        spacy_data = []

        # Add main spaCy directory
        spacy_data.append((spacy_dir, 'spacy'))

        # CRITICAL: Add Spanish model with MULTIPLE path strategies
        spacy_data.append((model_dir, 'es_core_news_sm'))

        # ADDITIONAL: Add model to spaCy's expected locations
        spacy_data.append((model_dir, 'spacy/data/es_core_news_sm'))
        spacy_data.append((model_dir, 'lib/python3.11/site-packages/es_core_news_sm'))

        # Add spaCy language data
        lang_dir = os.path.join(spacy_dir, 'lang')
        if os.path.exists(lang_dir):
            spacy_data.append((lang_dir, 'spacy/lang'))

        # Add spaCy pipeline components
        pipeline_dir = os.path.join(spacy_dir, 'pipeline')
        if os.path.exists(pipeline_dir):
            spacy_data.append((pipeline_dir, 'spacy/pipeline'))

        # Add spaCy training data
        training_dir = os.path.join(spacy_dir, 'training')
        if os.path.exists(training_dir):
            spacy_data.append((training_dir, 'spacy/training'))

        # Add spaCy lookups
        lookups_dir = os.path.join(spacy_dir, 'lookups')
        if os.path.exists(lookups_dir):
            spacy_data.append((lookups_dir, 'spacy/lookups'))

        print(f"✅ Collected {len(spacy_data)} spaCy data directories")
        return spacy_data

    except ImportError as e:
        print(f"❌ spaCy import error: {e}")
        return []

# COMPREHENSIVE binary collection for spaCy
def collect_spacy_binaries():
    """Collect all spaCy binary files"""
    try:
        import spacy
        spacy_dir = os.path.dirname(spacy.__file__)
        
        binaries = []
        
        # Find all .pyd files (Windows compiled extensions)
        pyd_files = glob.glob(os.path.join(spacy_dir, '**', '*.pyd'), recursive=True)
        for pyd_file in pyd_files:
            rel_path = os.path.relpath(pyd_file, spacy_dir)
            binaries.append((pyd_file, f'spacy/{os.path.dirname(rel_path)}'))
        
        # Find all .so files (Linux compiled extensions)
        so_files = glob.glob(os.path.join(spacy_dir, '**', '*.so'), recursive=True)
        for so_file in so_files:
            rel_path = os.path.relpath(so_file, spacy_dir)
            binaries.append((so_file, f'spacy/{os.path.dirname(rel_path)}'))
        
        print(f"✅ Collected {len(binaries)} spaCy binary files")
        return binaries
        
    except ImportError:
        print("❌ spaCy not available for binary collection")
        return []

# Collect spaCy data and binaries
spacy_datas = collect_spacy_data()
spacy_binaries = collect_spacy_binaries()

# Define all source files and directories to include
datas = [
    # Source Files directory (Excel files and database) - from CLIENT folder
    ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files', 'Source_Files'),

    # Models directory (all AI models) - from CLIENT folder
    ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/models', 'models'),
]

# Add spaCy data
datas.extend(spacy_datas)

# Define hidden imports - all modules that need to be explicitly included
hiddenimports = [
    # Core application modules
    'src.models',
    'src.models.sku_nn_pytorch',
    'src.utils.dummy_tokenizer',
    'src.utils.pytorch_tokenizer',
    'src.utils.logging_config',
    'src.utils.optimized_database',
    'src.utils.optimized_startup',
    'src.utils.year_range_database',
    'src.train_vin_predictor',
    'src.unified_consolidado_processor',
    'src.utils.performance_improvements',
    
    # GUI modules
    'tkinter.filedialog',
    'tkinter.font',
    
    # Data processing
    'numpy.core._multiarray_tests',
    'numpy.linalg.lapack_lite',
    'numpy._core',
    'numpy._core._dtype_ctypes',
    'numpy._core._multiarray_tests',
    'numpy._core._exceptions',
    'numpy._core._multiarray_umath',
    'pandas._libs.tslibs.base',
    'pandas._libs.reduction',
    'sklearn.utils._cython_blas',
    
    # COMPREHENSIVE spaCy imports
    'spacy',
    'spacy.cli',
    'spacy.lang',
    'spacy.lang.es',
    'spacy.lang.es.lex_attrs',
    'spacy.lang.es.punctuation',
    'spacy.lang.es.stop_words',
    'spacy.lang.es.syntax_iterators',
    'spacy.lang.es.tokenizer_exceptions',
    'spacy.pipeline',
    'spacy.pipeline.tok2vec',
    'spacy.pipeline.tagger',
    'spacy.pipeline.parser',
    'spacy.pipeline.ner',
    'spacy.pipeline.lemmatizer',
    'spacy.pipeline.morphologizer',
    'spacy.pipeline.attribute_ruler',
    'spacy.training',
    'spacy.lookups',
    'spacy.vectors',
    'spacy.vocab',
    'spacy.tokens',
    'spacy.matcher',
    'spacy.util',
    'es_core_news_sm',

    # Thinc (spaCy's ML library)
    'thinc',
    'thinc.api',
    'thinc.backends',
    'thinc.layers',
    'thinc.model',
    'thinc.optimizers',
    'thinc.schedules',
    'thinc.loss',

    # Additional spaCy dependencies
    'catalogue',
    'confection',
    'cymem',
    'murmurhash',
    'preshed',
    'spacy_legacy',
    'spacy_loggers',
    'srsly',
    'wasabi',
    'weasel',

    # COMPREHENSIVE SciPy imports - CRITICAL FIX
    'scipy',
    'scipy._lib',
    'scipy._lib.array_api_compat',
    'scipy._lib.array_api_compat.numpy',
    'scipy._lib.array_api_compat.numpy.fft',
    'scipy._lib.array_api_compat.numpy.linalg',
    'scipy.fft',
    'scipy.fft._basic',
    'scipy.fft._helper',
    'scipy.fft._pocketfft',
    'scipy.linalg',
    'scipy.linalg.blas',
    'scipy.linalg.lapack',
    'scipy.sparse',
    'scipy.sparse.linalg',
    'scipy.special',
    'scipy.stats',
    'scipy.optimize',
    'scipy.integrate',
    'scipy.interpolate',
    'scipy.signal',
    'scipy.ndimage',

    # CRITICAL: Add the missing array_api_compat module - NOW INSTALLED!
    'array_api_compat',
    'array_api_compat.numpy',
    'array_api_compat.numpy.fft',
    'array_api_compat.numpy.linalg',
    'array_api_compat.common',
    'array_api_compat.common._helpers',

    # EXCLUDE problematic sklearn.externals imports (these are incorrect)
    # The correct imports are scipy._lib.array_api_compat, not sklearn.externals
    
    # Excel processing
    'openpyxl.utils.dataframe',
    'openpyxl.utils.inference',
    
    # HTTP requests
    'requests.help',
    
    # Logging
    'logging.handlers',
]

# Comprehensive binaries list
binaries = spacy_binaries

a = Analysis(
    ['src/main_app.py'],
    pathex=[
        os.getcwd(),
        src_path,
        os.path.join(src_path, 'models'),
        os.path.join(src_path, 'utils'),
        os.path.join(src_path, 'core'),
        os.path.join(src_path, 'gui'),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_spacy_fix.py'],  # Add our custom spaCy fix
    excludes=[
        # Exclude problematic sklearn.externals modules (these are incorrect paths)
        'sklearn.externals.array_api_compat',
        'sklearn.externals.array_api_compat.numpy',
        'sklearn.externals.array_api_compat.numpy.fft',
        'sklearn.externals.array_api_compat.numpy.linalg',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Fixacar_SKU_Predictor_ULTIMATE_COMPREHENSIVE',
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
