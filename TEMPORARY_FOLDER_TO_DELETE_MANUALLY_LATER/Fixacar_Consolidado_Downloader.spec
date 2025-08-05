# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# Collect all critical dependencies (minimal for downloader)
requests_datas, requests_binaries, requests_hiddenimports = collect_all('requests')
urllib3_datas, urllib3_binaries, urllib3_hiddenimports = collect_all('urllib3')

a = Analysis(
    ['src/download_consolidado.py'],
    pathex=['src'],
    binaries=requests_binaries + urllib3_binaries,
    datas=[
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/Source_Files', 'Source_Files'),
        ('Fixacar_NUCLEAR_DEPLOYMENT/Fixacar_SKU_Predictor_CLIENT/logs', 'logs'),
    ] + requests_datas + urllib3_datas,
    hiddenimports=requests_hiddenimports + urllib3_hiddenimports + [
        # Force include ALL network dependencies
        'requests', 'urllib3', 'urllib', 'urllib.request', 'urllib.parse', 'urllib.error',
        'http.client', 'ssl', 'certifi', 'charset_normalizer',

        # System and utility libraries
        'json', 'logging', 'datetime', 'os', 'sys', 'shutil', 'pathlib',
        'collections', 'itertools', 'functools', 'operator',

        # Additional critical imports for standalone execution
        'pkg_resources', 'setuptools', 'distutils', 'importlib', 'importlib.util'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest', 'pandas', 'numpy', 'sklearn', 'torch'],
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
    name='1. Fixacar_Consolidado_Downloader',
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
