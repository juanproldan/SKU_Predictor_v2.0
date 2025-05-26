#!/usr/bin/env python3
"""
Simple test script to verify that all required packages can be imported.
"""

def test_imports():
    """Test importing all required packages."""
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
        print(f"   pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False

    try:
        import joblib
        print("✅ joblib imported successfully")
        print(f"   joblib version: {joblib.__version__}")
    except ImportError as e:
        print(f"❌ joblib import failed: {e}")
        return False

    try:
        import numpy as np
        print("✅ numpy imported successfully")
        print(f"   numpy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False

    try:
        import torch
        print("✅ torch imported successfully")
        print(f"   torch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False

    print("\n🎉 All imports successful!")
    return True

if __name__ == "__main__":
    test_imports()
