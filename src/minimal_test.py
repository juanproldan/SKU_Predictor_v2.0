#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MINIMAL TEST VERSION - Fixacar SKU Predictor
This version will help identify exactly where the application is getting stuck
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import traceback

print("🚀 MINIMAL TEST VERSION STARTING...")
print("=" * 50)

def test_imports():
    """Test all critical imports step by step"""
    print("📦 Testing imports...")
    
    try:
        print("  Testing basic Python modules...")
        import json, sqlite3, pickle, datetime
        print("  ✅ Basic Python modules OK")
        
        print("  Testing NumPy...")
        import numpy as np
        print(f"  ✅ NumPy {np.__version__} OK")
        
        print("  Testing Pandas...")
        import pandas as pd
        print(f"  ✅ Pandas {pd.__version__} OK")
        
        print("  Testing PyTorch...")
        import torch
        print(f"  ✅ PyTorch {torch.__version__} OK")
        
        print("  Testing scikit-learn...")
        import sklearn
        print(f"  ✅ Scikit-learn {sklearn.__version__} OK")
        
        print("  Testing openpyxl...")
        import openpyxl
        print(f"  ✅ OpenPyXL {openpyxl.__version__} OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_custom_imports():
    """Test custom module imports"""
    print("🔧 Testing custom imports...")
    
    try:
        print("  Testing utils...")
        from utils.dummy_tokenizer import DummyTokenizer
        print("  ✅ utils.dummy_tokenizer OK")
        
        print("  Testing models...")
        from models.sku_nn_pytorch import load_model
        print("  ✅ models.sku_nn_pytorch OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Custom import failed: {e}")
        traceback.print_exc()
        return False

def test_file_access():
    """Test file access"""
    print("📁 Testing file access...")
    
    current_dir = os.getcwd()
    print(f"  Current directory: {current_dir}")
    
    # Check for critical files
    files_to_check = [
        "Source_Files/Text_Processing_Rules.xlsx",
        "Source_Files/Maestro.xlsx", 
        "Source_Files/processed_consolidado.db"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)
            print(f"  ✅ {file_path}: {size:.1f} MB")
        else:
            print(f"  ⚠️ {file_path}: NOT FOUND")
    
    return True

def test_gui_creation():
    """Test GUI creation step by step"""
    print("🖥️ Testing GUI creation...")
    
    try:
        print("  Creating root window...")
        root = tk.Tk()
        print("  ✅ Root window created")
        
        print("  Setting window properties...")
        root.title("MINIMAL TEST - Fixacar SKU Predictor")
        root.geometry("800x600")
        print("  ✅ Window properties set")
        
        print("  Creating test widgets...")
        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        label = ttk.Label(frame, text="🎉 MINIMAL TEST SUCCESS!", font=("Arial", 16, "bold"))
        label.pack(pady=20)
        
        info_text = tk.Text(frame, height=15, width=80)
        info_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add test information
        test_info = """
MINIMAL TEST RESULTS:

✅ All imports working correctly
✅ File access working
✅ GUI creation successful

This means the core application should work!

The issue might be in:
1. Complex initialization code
2. Model loading
3. Database operations
4. Threading operations

Next steps:
1. If you see this window, the basic app works
2. We can gradually add more features
3. Identify exactly what causes the hang
"""
        
        info_text.insert(tk.END, test_info)
        info_text.config(state=tk.DISABLED)
        
        button = ttk.Button(frame, text="Close Test", command=root.quit)
        button.pack(pady=10)
        
        print("  ✅ Test widgets created")
        
        print("🚀 Starting GUI main loop...")
        print("=" * 50)
        print("GUI SHOULD NOW BE VISIBLE!")
        
        root.mainloop()
        
        print("✅ GUI closed normally")
        return True
        
    except Exception as e:
        print(f"❌ GUI creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 MINIMAL TEST VERSION")
    print("This version tests each component separately")
    print("=" * 50)
    
    try:
        # Test imports
        if not test_imports():
            print("❌ Import test failed")
            input("Press Enter to exit...")
            return False
        
        # Test custom imports
        if not test_custom_imports():
            print("❌ Custom import test failed")
            input("Press Enter to exit...")
            return False
        
        # Test file access
        test_file_access()
        
        # Test GUI
        if not test_gui_creation():
            print("❌ GUI test failed")
            input("Press Enter to exit...")
            return False
        
        print("✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
