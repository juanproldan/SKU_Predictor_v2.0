"""
PyInstaller hook for scipy
Ensures all scipy submodules are properly included
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all scipy submodules
hiddenimports = collect_submodules('scipy')

# Add specific problematic modules
hiddenimports += [
    'scipy._lib.array_api_compat.numpy.fft',
    'scipy._lib.array_api_compat.numpy',
    'scipy._lib.array_api_compat',
    'scipy._lib._array_api',
    'scipy._lib._util',
    'scipy.sparse._base',
    'scipy.sparse._sputils',
    'scipy.special._ufuncs',
    'scipy.special._ufuncs_cxx',
    'scipy.linalg._fblas',
    'scipy.linalg._flapack',
    'scipy.linalg.cython_blas',
    'scipy.linalg.cython_lapack',
]

# Collect data files
datas = collect_data_files('scipy')
