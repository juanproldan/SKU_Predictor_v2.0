# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec file for Consolidado.json Downloader
Creates a standalone executable for automated data downloads
"""

import sys
from pathlib import Path

# Get the current directory
current_dir = Path.cwd()

block_cipher = None

a = Analysis(
    ['src/download_consolidado.py'],
    pathex=[str(current_dir)],
    binaries=[],
    datas=[],
    hiddenimports=[
        'requests',
        'json',
        'datetime',
        'logging',
        'pathlib',
        'shutil',
        'tempfile',
        'os',
        'sys'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'torch',
        'torchvision',
        'PIL',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6'
    ],
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
    name='Fixacar_Consolidado_Downloader',
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
    icon=None,
    version_info=None
)
