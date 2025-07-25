# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec file for Unified Consolidado Processor
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(SPEC))

# Define paths
src_path = os.path.join(current_dir, 'src')
data_path = os.path.join(current_dir, 'data')
source_files_path = os.path.join(current_dir, 'Source_Files')

# Collect all submodules for problematic packages
pandas_submodules = collect_submodules('pandas')
numpy_submodules = collect_submodules('numpy')
openpyxl_submodules = collect_submodules('openpyxl')

# Collect data files for packages that need them
pandas_data = collect_data_files('pandas')
openpyxl_data = collect_data_files('openpyxl')

# Define hidden imports
hidden_imports = [
    # Core Python libraries
    'os',
    'sys',
    'json',
    'sqlite3',
    'logging',
    're',
    'datetime',
    'pathlib',
    
    # Data processing
    'pandas',
    'pandas._libs.tslibs.base',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.timestamps',
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.lib.format',
    
    # Excel processing
    'openpyxl',
    'openpyxl.workbook',
    'openpyxl.worksheet',
    'openpyxl.reader',
    'openpyxl.writer',
    
    # Text processing utilities
    'src.utils.text_utils',
]

# Add collected submodules
hidden_imports.extend(pandas_submodules)
hidden_imports.extend(numpy_submodules)
hidden_imports.extend(openpyxl_submodules)

# Remove duplicates
hidden_imports = list(set(hidden_imports))

# Define data files to include
datas = [
    (source_files_path, 'Source_Files'),
    (src_path, 'src'),
]

# Add collected data files
datas.extend(pandas_data)
datas.extend(openpyxl_data)

# Analysis configuration
a = Analysis(
    ['src/unified_consolidado_processor.py'],
    pathex=[current_dir, src_path],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'setuptools',
        'torch',
        'sklearn',
        'scipy',
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
    name='Fixacar_Consolidado_Processor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Console application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
