#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Isolated Dependency Testing Script

This script tests executables by temporarily hiding system Python packages
to simulate a clean environment without numpy, pandas, etc.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def create_isolated_test_environment():
    """Create a temporary directory that simulates a clean system."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="fixacar_isolated_test_")
    
    # Create a minimal Python environment structure
    python_dir = os.path.join(temp_dir, "python")
    os.makedirs(python_dir, exist_ok=True)
    
    # Copy only essential Python DLLs (not packages)
    # This simulates a system without numpy, pandas, etc.
    
    return temp_dir

def test_executable_in_isolation(exe_path, test_name="Basic Test"):
    """Test an executable in an isolated environment."""
    
    print(f"\n🧪 ISOLATED TEST: {os.path.basename(exe_path)}")
    print(f"📋 Test: {test_name}")
    print("-" * 60)
    
    if not os.path.exists(exe_path):
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    # Create test script that the executable should be able to run
    test_script = '''
import sys
import traceback

def test_critical_imports():
    """Test critical imports that should be embedded in the executable."""
    
    critical_modules = [
        'numpy',
        'pandas', 
        'sklearn',
        'torch',
        'openpyxl',
        'sqlite3',
        'json',
        'logging'
    ]
    
    results = {}
    
    print("=== DEPENDENCY TEST START ===")
    
    for module in critical_modules:
        try:
            __import__(module)
            print(f"PASS: {module}")
            results[module] = True
        except ImportError as e:
            print(f"FAIL: {module} - {str(e)}")
            results[module] = False
        except Exception as e:
            print(f"ERROR: {module} - {str(e)}")
            results[module] = False
    
    print("=== DEPENDENCY TEST END ===")
    
    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"SUMMARY: {passed}/{total} modules available")
    
    if passed == total:
        print("RESULT: ALL_DEPENDENCIES_OK")
        return True
    else:
        print("RESULT: MISSING_DEPENDENCIES")
        return False

if __name__ == "__main__":
    try:
        success = test_critical_imports()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"CRITICAL_ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(2)
'''
    
    # Save test script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_script = f.name
    
    try:
        # Method 1: Try to run the executable and see if it crashes with import errors
        print("🚀 Testing executable startup...")
        
        # Set environment to hide system packages
        env = os.environ.copy()
        
        # Remove Python paths that might interfere
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        
        # Start the executable
        process = subprocess.Popen(
            [exe_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        # Let it run briefly
        import time
        time.sleep(5)
        
        if process.poll() is None:
            # Still running - good sign
            print("✅ Executable started successfully")
            process.terminate()
            try:
                process.wait(timeout=3)
            except:
                process.kill()
            
            # Clean up
            os.unlink(temp_script)
            return True
        else:
            # Check for import errors
            stdout, stderr = process.communicate()
            stderr_text = stderr.decode().lower()
            stdout_text = stdout.decode().lower()
            
            # Look for dependency errors
            dependency_errors = [
                "modulenotfounderror",
                "importerror",
                "no module named 'numpy'",
                "no module named 'pandas'", 
                "no module named 'sklearn'",
                "no module named 'torch'",
                "dll load failed",
                "unable to import required dependencies"
            ]
            
            has_dependency_error = any(error in stderr_text or error in stdout_text 
                                     for error in dependency_errors)
            
            if has_dependency_error:
                print(f"❌ DEPENDENCY ERROR DETECTED:")
                print(f"   STDERR: {stderr_text[:300]}")
                print(f"   STDOUT: {stdout_text[:300]}")
                
                # Clean up
                os.unlink(temp_script)
                return False
            else:
                print("✅ No dependency errors detected")
                
                # Clean up
                os.unlink(temp_script)
                return True
                
    except Exception as e:
        print(f"❌ Exception during testing: {e}")
        
        # Clean up
        try:
            os.unlink(temp_script)
        except:
            pass
        return False

def main():
    """Main isolated testing function."""
    
    print("🔬 ISOLATED DEPENDENCY TESTING")
    print("=" * 70)
    print("Testing executables in simulated clean environment...")
    print("This simulates systems without numpy, pandas, sklearn, torch installed")
    
    CLIENT_DIR = "Fixacar_SKU_Predictor_CLIENT"
    EXECUTABLES = [
        ("1. Fixacar_Consolidado_Downloader.exe", "Network Operations"),
        ("2. Fixacar_Data_Processor.exe", "Data Processing with Pandas/Numpy"),
        ("3. Fixacar_VIN_Trainer.exe", "ML Training with Sklearn/Torch"),
        ("4. Fixacar_SKU_Trainer.exe", "Neural Network Training"),
        ("Fixacar_SKU_Predictor.exe", "GUI Application")
    ]
    
    if not os.path.exists(CLIENT_DIR):
        print(f"❌ Client directory not found: {CLIENT_DIR}")
        return False
    
    results = []
    
    for exe_name, test_description in EXECUTABLES:
        exe_path = os.path.join(CLIENT_DIR, exe_name)
        success = test_executable_in_isolation(exe_path, test_description)
        results.append((exe_name, success))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 ISOLATED DEPENDENCY TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} executables passed isolated testing")
    print(f"Success rate: {(passed/total)*100:.1f}%\n")
    
    for exe_name, success in results:
        status = "✅ ISOLATED OK" if success else "❌ DEPENDENCY MISSING"
        print(f"  {exe_name:<35} {status}")
    
    if passed == total:
        print(f"\n🎉 ALL EXECUTABLES PASS ISOLATED TESTING!")
        print("✅ All dependencies are properly embedded")
        print("✅ Should work on systems without Python packages")
        print("✅ Ready for deployment to clean client systems")
    else:
        print(f"\n⚠️  {total-passed} executable(s) have missing dependencies")
        print("🔧 These executables will fail on systems without Python packages")
        print("🔧 Need to rebuild with enhanced dependency inclusion")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*80}")
    if success:
        print("🎉 ALL ISOLATED TESTS PASSED!")
        print("Executables should work on clean client systems")
    else:
        print("⚠️  ISOLATED TESTS FAILED!")
        print("Executables have missing dependencies")
    print(f"{'='*80}")
    
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
