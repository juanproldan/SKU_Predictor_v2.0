#!/usr/bin/env python3
"""
Dependency Verification Script for Fixacar SKU Predictor
This script checks if all required dependencies are properly installed
and can be imported without errors.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Python Version Check:")
    print(f"   Version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("   ❌ ERROR: Python 3.8+ required")
        return False
    else:
        print("   ✅ Python version OK")
        return True

def check_virtual_env():
    """Check if virtual environment is active."""
    print("\n🏠 Virtual Environment Check:")
    
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print("   ✅ Virtual environment is active")
        print(f"   Path: {sys.prefix}")
    else:
        print("   ⚠️  No virtual environment detected")
        print("   Recommendation: Use virtual environment for better isolation")
    
    return True

def check_import(module_name, description=""):
    """Check if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"   ✅ {module_name} ({version}) {description}")
        return True
    except ImportError as e:
        print(f"   ❌ {module_name} - MISSING: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️  {module_name} - WARNING: {e}")
        return False

def check_core_dependencies():
    """Check core Python dependencies."""
    print("\n📦 Core Dependencies Check:")
    
    dependencies = [
        ('numpy', '- Numerical computing'),
        ('pandas', '- Data manipulation'),
        ('sqlite3', '- Database operations'),
        ('tkinter', '- GUI framework'),
        ('json', '- JSON processing'),
        ('re', '- Regular expressions'),
        ('os', '- Operating system interface'),
        ('datetime', '- Date/time handling'),
        ('collections', '- Data structures'),
    ]
    
    success_count = 0
    for module, desc in dependencies:
        if check_import(module, desc):
            success_count += 1
    
    return success_count == len(dependencies)

def check_ml_dependencies():
    """Check machine learning dependencies."""
    print("\n🤖 Machine Learning Dependencies Check:")
    
    dependencies = [
        ('sklearn', '- Scikit-learn'),
        ('joblib', '- Model serialization'),
        ('torch', '- PyTorch deep learning'),
        ('numpy', '- Already checked above'),
    ]
    
    success_count = 0
    for module, desc in dependencies:
        if check_import(module, desc):
            success_count += 1
    
    # Check specific sklearn submodules
    sklearn_modules = [
        'sklearn.preprocessing',
        'sklearn.model_selection', 
        'sklearn.metrics',
        'sklearn.naive_bayes',
    ]
    
    print("   Checking sklearn submodules:")
    for module in sklearn_modules:
        check_import(module)
    
    # Check specific torch submodules
    torch_modules = [
        'torch.nn',
        'torch.optim',
        'torch.utils.data',
    ]
    
    print("   Checking torch submodules:")
    for module in torch_modules:
        check_import(module)
    
    return success_count >= 3  # Allow some flexibility

def check_text_processing():
    """Check text processing dependencies."""
    print("\n📝 Text Processing Dependencies Check:")
    
    dependencies = [
        ('fuzzywuzzy', '- Fuzzy string matching'),
        ('Levenshtein', '- Fast string distance'),
        ('openpyxl', '- Excel file handling'),
    ]
    
    success_count = 0
    for module, desc in dependencies:
        if check_import(module, desc):
            success_count += 1
    
    return success_count >= 2  # fuzzywuzzy and openpyxl are critical

def check_custom_modules():
    """Check if custom modules can be imported."""
    print("\n🔧 Custom Modules Check:")
    
    # Add src to path temporarily
    src_path = Path(__file__).parent / 'src'
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    custom_modules = [
        ('utils.text_utils', '- Text normalization utilities'),
        ('utils.dummy_tokenizer', '- Tokenizer fallback'),
        ('models.sku_nn_pytorch', '- SKU neural network model'),
    ]
    
    success_count = 0
    for module, desc in custom_modules:
        if check_import(module, desc):
            success_count += 1
    
    return success_count >= 2  # Allow some flexibility

def check_data_files():
    """Check if required data files exist."""
    print("\n📁 Data Files Check:")
    
    required_files = [
        'data/',
        'models/',
        'Source_Files/',
        'src/main_app.py',
        'requirements.txt',
    ]
    
    success_count = 0
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
            success_count += 1
        else:
            print(f"   ❌ {file_path} - MISSING")
    
    return success_count == len(required_files)

def check_pyinstaller():
    """Check PyInstaller installation."""
    print("\n🔨 PyInstaller Check:")
    
    try:
        import PyInstaller
        print(f"   ✅ PyInstaller ({PyInstaller.__version__})")
        return True
    except ImportError:
        print("   ❌ PyInstaller not installed")
        print("   Run: pip install pyinstaller")
        return False

def main():
    """Run all dependency checks."""
    print("=" * 60)
    print("🔍 FIXACAR SKU PREDICTOR - DEPENDENCY VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Core Dependencies", check_core_dependencies),
        ("ML Dependencies", check_ml_dependencies),
        ("Text Processing", check_text_processing),
        ("Custom Modules", check_custom_modules),
        ("Data Files", check_data_files),
        ("PyInstaller", check_pyinstaller),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 All checks passed! Ready to build executables.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} issues found. Please fix before building.")
        print("\nRecommended actions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure all data files are present")
        print("3. Check Python version compatibility")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
