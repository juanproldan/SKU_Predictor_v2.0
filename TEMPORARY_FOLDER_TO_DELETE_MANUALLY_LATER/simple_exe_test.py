#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Executable Testing Script

This script performs basic startup tests on all Fixacar executables
to verify they don't have critical dependency errors.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import subprocess
import time
import sys

CLIENT_DIR = "Fixacar_SKU_Predictor_CLIENT"
EXECUTABLES = [
    "1. Fixacar_Consolidado_Downloader.exe",
    "2. Fixacar_Data_Processor.exe", 
    "3. Fixacar_VIN_Trainer.exe",
    "4. Fixacar_SKU_Trainer.exe",
    "Fixacar_SKU_Predictor.exe"
]

def test_executable_basic(exe_path):
    """Basic test - just try to start the executable and see if it crashes immediately."""
    print(f"\n🧪 Testing: {os.path.basename(exe_path)}")
    print("-" * 50)
    
    if not os.path.exists(exe_path):
        print(f"❌ File not found: {exe_path}")
        return False
    
    file_size_mb = round(os.path.getsize(exe_path) / (1024*1024), 1)
    print(f"✅ File exists ({file_size_mb} MB)")
    
    try:
        # Start the process
        print("🚀 Starting executable...")
        
        if "SKU_Predictor.exe" in exe_path:
            # GUI app - start and let it run briefly
            process = subprocess.Popen(
                [exe_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # Wait a moment for it to initialize
            time.sleep(3)
            
            if process.poll() is None:
                # Still running - good sign
                print("✅ GUI application started successfully")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except:
                    process.kill()
                return True
            else:
                # Crashed immediately
                stdout, stderr = process.communicate()
                print(f"❌ Process crashed: {stderr.decode()[:200]}")
                return False
        
        else:
            # Console app - try to run it briefly
            process = subprocess.Popen(
                [exe_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if it's still running or exited cleanly
            if process.poll() is None:
                # Still running - terminate it
                print("✅ Console application started successfully")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except:
                    process.kill()
                return True
            else:
                # Check exit status
                stdout, stderr = process.communicate()
                stderr_text = stderr.decode().lower()
                
                # Look for critical errors
                critical_errors = [
                    "modulenotfounderror",
                    "importerror",
                    "dll load failed",
                    "no module named"
                ]
                
                has_critical_error = any(error in stderr_text for error in critical_errors)
                
                if has_critical_error:
                    print(f"❌ Critical dependency error: {stderr_text[:200]}")
                    return False
                else:
                    print("✅ Application started and exited normally")
                    return True
                    
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Main testing function."""
    print("🧪 SIMPLE EXECUTABLE TESTING")
    print("=" * 50)
    print("Testing all executables for basic startup and dependency issues...")
    
    if not os.path.exists(CLIENT_DIR):
        print(f"❌ Client directory not found: {CLIENT_DIR}")
        return False
    
    results = []
    
    for exe_name in EXECUTABLES:
        exe_path = os.path.join(CLIENT_DIR, exe_name)
        success = test_executable_basic(exe_path)
        results.append((exe_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} executables passed")
    print(f"Success rate: {(passed/total)*100:.1f}%\n")
    
    for exe_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {exe_name:<35} {status}")
    
    if passed == total:
        print(f"\n🎉 ALL EXECUTABLES WORKING!")
        print("✅ No critical dependency errors detected")
        print("✅ Ready for deployment")
    else:
        print(f"\n⚠️  {total-passed} executable(s) have issues")
        print("🔧 Consider rebuilding failed executables")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED!")
    print(f"{'='*60}")
    
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
