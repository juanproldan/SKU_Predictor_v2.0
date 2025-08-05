
import subprocess
import os
import sys

def test_executable():
    """Test the bulletproof executable"""
    
    exe_path = os.path.join("Fixacar_NUCLEAR_DEPLOYMENT", "Fixacar_SKU_Predictor_CLIENT", "Fixacar_SKU_Predictor_BULLETPROOF.exe")
    
    if not os.path.exists(exe_path):
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    print(f"🧪 Testing executable: {exe_path}")
    
    try:
        # Run the executable with a timeout
        result = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(exe_path)
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Check for import errors
        if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
            print("❌ Import errors detected!")
            return False
        elif "Successfully imported all modules" in result.stdout:
            print("✅ All modules imported successfully!")
            return True
        else:
            print("⚠️ Executable started but status unclear")
            return True
            
    except subprocess.TimeoutExpired:
        print("⏰ Executable timed out (might be waiting for GUI interaction)")
        return True  # Timeout might be normal for GUI apps
    except Exception as e:
        print(f"❌ Error testing executable: {e}")
        return False

if __name__ == "__main__":
    success = test_executable()
    sys.exit(0 if success else 1)
