#!/usr/bin/env python3
"""
Executable Testing Script for Fixacar SKU Predictor
This script tests the built executables to ensure they work correctly
before deployment to client.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_executable_exists(exe_path):
    """Check if executable exists."""
    path = Path(exe_path)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Found: {exe_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"   ❌ Missing: {exe_path}")
        return False

def test_executable_launch(exe_path, timeout=10):
    """Test if executable can launch without immediate crashes."""
    print(f"\n🚀 Testing launch: {exe_path}")
    
    try:
        # Start the process
        process = subprocess.Popen(
            [exe_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit to see if it crashes immediately
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("   ✅ Process launched successfully and is running")
            
            # For GUI apps, we can't easily test further without user interaction
            # For console apps, we might get some output
            try:
                # Try to get some output (non-blocking)
                stdout, stderr = process.communicate(timeout=2)
                if stdout:
                    print(f"   📄 Output preview: {stdout[:200]}...")
                if stderr and "error" in stderr.lower():
                    print(f"   ⚠️  Stderr: {stderr[:200]}...")
            except subprocess.TimeoutExpired:
                # This is expected for GUI apps
                pass
            
            # Terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            return True
        else:
            # Process crashed
            stdout, stderr = process.communicate()
            print("   ❌ Process crashed immediately")
            if stderr:
                print(f"   Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to launch: {e}")
        return False

def test_sku_predictor_gui():
    """Test the main SKU Predictor GUI."""
    print("\n" + "="*50)
    print("🖥️  TESTING SKU PREDICTOR GUI")
    print("="*50)
    
    exe_path = "dist/Fixacar_SKU_Predictor.exe"
    
    if not check_executable_exists(exe_path):
        return False
    
    print("\n📋 Manual Test Instructions:")
    print("   The GUI will launch for 10 seconds.")
    print("   Please verify:")
    print("   1. Window opens without errors")
    print("   2. All UI elements are visible")
    print("   3. No immediate error popups")
    print("   4. Application responds to interaction")
    
    input("\nPress Enter to launch GUI test...")
    
    return test_executable_launch(exe_path, timeout=10)

def test_vin_trainer():
    """Test the VIN Trainer."""
    print("\n" + "="*50)
    print("🔧 TESTING VIN TRAINER")
    print("="*50)
    
    exe_path = "dist/Fixacar_VIN_Trainer.exe"
    
    if not check_executable_exists(exe_path):
        return False
    
    print("\n⚠️  Note: VIN Trainer will run briefly to test imports")
    print("   It may show 'No data found' - this is expected in test mode")
    
    return test_executable_launch(exe_path, timeout=15)

def test_sku_trainer():
    """Test the SKU Trainer."""
    print("\n" + "="*50)
    print("🧠 TESTING SKU TRAINER")
    print("="*50)
    
    exe_path = "dist/Fixacar_SKU_Trainer.exe"
    
    if not check_executable_exists(exe_path):
        return False
    
    print("\n⚠️  Note: SKU Trainer will run briefly to test imports")
    print("   It may show 'No data found' - this is expected in test mode")
    
    return test_executable_launch(exe_path, timeout=15)

def check_dependencies_bundled():
    """Check if critical dependencies are bundled."""
    print("\n" + "="*50)
    print("📦 CHECKING BUNDLED DEPENDENCIES")
    print("="*50)
    
    # Check if dist folder has reasonable size
    dist_path = Path("dist")
    if not dist_path.exists():
        print("   ❌ dist/ folder not found")
        return False
    
    total_size = 0
    exe_count = 0
    
    for exe_file in dist_path.glob("*.exe"):
        size = exe_file.stat().st_size
        size_mb = size / (1024 * 1024)
        total_size += size
        exe_count += 1
        print(f"   📁 {exe_file.name}: {size_mb:.1f} MB")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n   📊 Total size: {total_mb:.1f} MB ({exe_count} executables)")
    
    # Reasonable size check (PyTorch + sklearn + pandas should be substantial)
    if total_mb < 100:
        print("   ⚠️  WARNING: Total size seems small - dependencies might be missing")
        return False
    elif total_mb > 2000:
        print("   ⚠️  WARNING: Total size is very large - might include unnecessary files")
    else:
        print("   ✅ Size looks reasonable for bundled ML dependencies")
    
    return True

def create_deployment_checklist():
    """Create a deployment checklist file."""
    checklist = """
# 🚀 FIXACAR SKU PREDICTOR - DEPLOYMENT CHECKLIST

## Pre-Deployment Testing (COMPLETED ✅)
- [ ] All executables built successfully
- [ ] GUI launches without errors
- [ ] Trainers can start without import errors
- [ ] File sizes are reasonable
- [ ] Dependencies properly bundled

## Client Deployment Steps
1. **Copy Files to Client**
   - Copy entire `dist/` folder to client laptop
   - Recommended location: `C:\\Fixacar\\`

2. **Test on Client Machine**
   - Run `Fixacar_SKU_Predictor.exe` 
   - Verify GUI opens and functions work
   - Test with sample data if available

3. **Setup Automation (Windows Task Scheduler)**
   - Weekly VIN Training: `Fixacar_VIN_Trainer.exe`
   - Monthly SKU Training: `Fixacar_SKU_Trainer.exe`
   - Schedule during off-hours (e.g., 2 AM)

4. **Data Management**
   - Ensure `data/` folder has required files
   - Verify `Source_Files/` contains Excel files
   - Check `models/` folder for trained models

## Troubleshooting
- If "numpy import error": Rebuild with improved script
- If "missing DLL": Install Visual C++ Redistributable
- If slow startup: Normal for first run (dependency loading)

## Support Contact
- Developer: [Your contact information]
- Last Updated: {timestamp}
"""
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checklist = checklist.format(timestamp=timestamp)
    
    with open("DEPLOYMENT_CHECKLIST.md", "w", encoding="utf-8") as f:
        f.write(checklist)
    
    print("   ✅ Created DEPLOYMENT_CHECKLIST.md")

def main():
    """Run all executable tests."""
    print("=" * 60)
    print("🧪 FIXACAR SKU PREDICTOR - EXECUTABLE TESTING")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("dist").exists():
        print("❌ ERROR: dist/ folder not found")
        print("Please run this script from the project root after building executables")
        return False
    
    tests = [
        ("Dependency Bundling", check_dependencies_bundled),
        ("SKU Predictor GUI", test_sku_predictor_gui),
        ("VIN Trainer", test_vin_trainer),
        ("SKU Trainer", test_sku_trainer),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Executables are ready for deployment.")
        create_deployment_checklist()
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed.")
        print("Please fix issues before deploying to client.")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
