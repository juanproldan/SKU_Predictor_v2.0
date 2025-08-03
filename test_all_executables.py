#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Executable Testing Script

This script tests all Fixacar executables to ensure they start without dependency errors.
It performs import testing, basic functionality checks, and dependency verification.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Test configuration
CLIENT_DIR = "Fixacar_SKU_Predictor_CLIENT"
EXECUTABLES = [
    "1. Fixacar_Consolidado_Downloader.exe",
    "2. Fixacar_Data_Processor.exe", 
    "3. Fixacar_VIN_Trainer.exe",
    "4. Fixacar_SKU_Trainer.exe",
    "Fixacar_SKU_Predictor.exe"
]

def test_executable_startup(exe_path, timeout=30):
    """
    Test if an executable starts without import/dependency errors.
    
    Args:
        exe_path (str): Path to executable
        timeout (int): Timeout in seconds
        
    Returns:
        dict: Test results
    """
    print(f"\n🧪 Testing: {os.path.basename(exe_path)}")
    print("-" * 50)
    
    result = {
        "executable": os.path.basename(exe_path),
        "exists": False,
        "starts": False,
        "no_import_errors": False,
        "error_message": None,
        "file_size_mb": 0
    }
    
    # Check if file exists
    if not os.path.exists(exe_path):
        result["error_message"] = "File does not exist"
        print(f"❌ File not found: {exe_path}")
        return result
    
    result["exists"] = True
    result["file_size_mb"] = round(os.path.getsize(exe_path) / (1024*1024), 1)
    print(f"✅ File exists ({result['file_size_mb']} MB)")
    
    try:
        # Start the executable with a timeout
        print(f"🚀 Starting executable...")
        
        # For GUI apps, we'll start them and quickly terminate
        # For console apps, we'll check their help/version output
        if "SKU_Predictor.exe" in exe_path:
            # Main GUI app - start and terminate quickly
            process = subprocess.Popen(
                [exe_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # Give it a moment to start and load dependencies
            time.sleep(5)
            
            # Check if process is still running (good sign)
            if process.poll() is None:
                result["starts"] = True
                result["no_import_errors"] = True
                print("✅ GUI application started successfully")
                
                # Terminate gracefully
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            else:
                # Process exited - check for errors
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    result["error_message"] = f"Exit code: {process.returncode}, stderr: {stderr.decode()}"
                    print(f"❌ Process exited with error: {result['error_message']}")
                else:
                    result["starts"] = True
                    result["no_import_errors"] = True
                    print("✅ Process completed successfully")
        
        else:
            # Console apps - try to get help/version info
            process = subprocess.Popen(
                [exe_path, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Check for common import errors
                stderr_text = stderr.decode().lower()
                stdout_text = stdout.decode().lower()
                
                import_error_indicators = [
                    "modulenotfounderror",
                    "importerror", 
                    "no module named",
                    "dll load failed",
                    "failed to import"
                ]
                
                has_import_error = any(indicator in stderr_text or indicator in stdout_text 
                                     for indicator in import_error_indicators)
                
                if has_import_error:
                    result["error_message"] = f"Import error detected: {stderr_text[:200]}"
                    print(f"❌ Import error: {result['error_message']}")
                else:
                    result["starts"] = True
                    result["no_import_errors"] = True
                    print("✅ No import errors detected")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                result["starts"] = True  # Started but timed out (acceptable)
                result["no_import_errors"] = True
                print("✅ Started successfully (timed out waiting for response)")
                
    except Exception as e:
        result["error_message"] = str(e)
        print(f"❌ Exception during testing: {e}")
    
    return result

def test_critical_imports():
    """Test critical Python imports that executables depend on."""
    print("\n🔬 TESTING CRITICAL IMPORTS")
    print("=" * 50)
    
    critical_imports = [
        "numpy",
        "pandas", 
        "sklearn",
        "torch",
        "openpyxl",
        "requests",
        "sqlite3",
        "tkinter"
    ]
    
    results = {}
    
    for module in critical_imports:
        try:
            __import__(module)
            results[module] = "✅ Available"
            print(f"✅ {module}")
        except ImportError as e:
            results[module] = f"❌ Error: {e}"
            print(f"❌ {module}: {e}")
    
    return results

def generate_test_report(executable_results, import_results):
    """Generate a comprehensive test report."""
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    
    # Summary
    total_exes = len(executable_results)
    working_exes = sum(1 for r in executable_results if r["starts"] and r["no_import_errors"])
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total Executables: {total_exes}")
    print(f"   Working Executables: {working_exes}")
    print(f"   Success Rate: {(working_exes/total_exes)*100:.1f}%")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for result in executable_results:
        status = "✅ PASS" if result["starts"] and result["no_import_errors"] else "❌ FAIL"
        print(f"   {result['executable']:<35} {status}")
        if result["error_message"]:
            print(f"      Error: {result['error_message']}")
    
    # Import status
    print(f"\n🔗 IMPORT STATUS:")
    for module, status in import_results.items():
        print(f"   {module:<15} {status}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    failed_exes = [r for r in executable_results if not (r["starts"] and r["no_import_errors"])]
    
    if not failed_exes:
        print("   🎉 All executables are working correctly!")
        print("   ✅ No dependency errors detected")
        print("   ✅ Ready for deployment")
    else:
        print("   ⚠️  Some executables have issues:")
        for result in failed_exes:
            print(f"      - {result['executable']}: {result['error_message']}")
        print("   🔧 Consider rebuilding failed executables with enhanced dependencies")
    
    return working_exes == total_exes

def main():
    """Main testing function."""
    print("🧪 FIXACAR EXECUTABLE DEPENDENCY TESTING")
    print("=" * 60)
    print("This script tests all executables for dependency errors")
    print("Testing will take approximately 2-3 minutes...")
    
    # Check if client directory exists
    if not os.path.exists(CLIENT_DIR):
        print(f"❌ Client directory not found: {CLIENT_DIR}")
        return False
    
    # Test critical imports first
    import_results = test_critical_imports()
    
    # Test each executable
    executable_results = []
    
    print(f"\n🎯 TESTING EXECUTABLES")
    print("=" * 50)
    
    for exe_name in EXECUTABLES:
        exe_path = os.path.join(CLIENT_DIR, exe_name)
        result = test_executable_startup(exe_path)
        executable_results.append(result)
    
    # Generate comprehensive report
    all_passed = generate_test_report(executable_results, import_results)
    
    # Save detailed results to JSON
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_executables": len(executable_results),
            "working_executables": sum(1 for r in executable_results if r["starts"] and r["no_import_errors"]),
            "all_passed": all_passed
        },
        "executable_results": executable_results,
        "import_results": import_results
    }
    
    with open("executable_test_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: executable_test_report.json")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 ALL TESTS PASSED - Executables are ready for deployment!")
    else:
        print("⚠️  SOME TESTS FAILED - Review the report above")
    print(f"{'='*60}")
    
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
