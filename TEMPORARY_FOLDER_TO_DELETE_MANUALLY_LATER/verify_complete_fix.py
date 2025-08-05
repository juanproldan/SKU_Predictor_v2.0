#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Verification of VIN Prediction Fix

This script verifies that all file naming inconsistencies have been resolved
and that the VIN prediction system is working correctly.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def verify_model_files():
    """Verify that all model files exist with correct naming."""
    
    print("🔍 VERIFYING MODEL FILE CONSISTENCY")
    print("=" * 50)
    
    try:
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        model_dir = os.path.join(base_path, "models")
        
        print(f"📂 Model directory: {model_dir}")
        
        # Expected files with correct naming
        expected_files = [
            'maker_model.joblib',
            'maker_encoder_x.joblib', 
            'maker_encoder_y.joblib',
            'model_model.joblib',
            'model_encoder_x.joblib',
            'model_encoder_y.joblib',
            'series_model.joblib',
            'series_encoder_x.joblib',
            'series_encoder_y.joblib'
        ]
        
        # Files that should NOT exist (old inconsistent naming)
        forbidden_files = [
            'makerr_model.joblib',
            'makerr_encoder_x.joblib',
            'makerr_encoder_y.joblib'
        ]
        
        print("\n✅ CHECKING EXPECTED FILES:")
        all_expected_exist = True
        for file in expected_files:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} - MISSING!")
                all_expected_exist = False
        
        print("\n🚫 CHECKING FORBIDDEN FILES:")
        no_forbidden_exist = True
        for file in forbidden_files:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                print(f"   ❌ {file} - SHOULD NOT EXIST!")
                no_forbidden_exist = False
            else:
                print(f"   ✅ {file} - Correctly absent")
        
        # List all actual files
        print(f"\n📋 ALL FILES IN {model_dir}:")
        if os.path.exists(model_dir):
            all_files = []
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), model_dir)
                    all_files.append(rel_path)
            
            for file in sorted(all_files):
                print(f"   📄 {file}")
        else:
            print("   ❌ Model directory does not exist!")
            return False
        
        return all_expected_exist and no_forbidden_exist
        
    except Exception as e:
        print(f"❌ Error verifying model files: {e}")
        return False

def verify_code_consistency():
    """Verify that all code references use consistent naming."""
    
    print("\n🔍 VERIFYING CODE CONSISTENCY")
    print("=" * 50)
    
    # Files to check for consistency
    files_to_check = [
        'src/train_vin_predictor.py',
        'src/train_sku_nn_predictor_pytorch_optimized.py', 
        'src/main_app.py',
        'apply_vin_prediction_fix.py'
    ]
    
    inconsistent_references = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"📄 Checking {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for old inconsistent naming
                if 'makerr_model.joblib' in content:
                    inconsistent_references.append(f"{file_path}: contains 'makerr_model.joblib'")
                if 'makerr_encoder' in content:
                    inconsistent_references.append(f"{file_path}: contains 'makerr_encoder'")
                if 'model_makerr' in content and 'model_maker' not in content.replace('model_makerr', ''):
                    inconsistent_references.append(f"{file_path}: uses 'model_makerr' variable")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    if inconsistent_references:
        print("\n❌ INCONSISTENT REFERENCES FOUND:")
        for ref in inconsistent_references:
            print(f"   ❌ {ref}")
        return False
    else:
        print("\n✅ ALL CODE REFERENCES ARE CONSISTENT!")
        return True

def test_model_loading():
    """Test that models can be loaded successfully."""
    
    print("\n🧪 TESTING MODEL LOADING")
    print("=" * 50)
    
    try:
        import joblib
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        model_dir = os.path.join(base_path, "models")
        
        # Test loading each model
        models_to_test = [
            ('Maker Model', 'maker_model.joblib'),
            ('Maker X Encoder', 'maker_encoder_x.joblib'),
            ('Maker Y Encoder', 'maker_encoder_y.joblib'),
            ('Model Model', 'model_model.joblib'),
            ('Model X Encoder', 'model_encoder_x.joblib'),
            ('Model Y Encoder', 'model_encoder_y.joblib'),
            ('Series Model', 'series_model.joblib'),
            ('Series X Encoder', 'series_encoder_x.joblib'),
            ('Series Y Encoder', 'series_encoder_y.joblib')
        ]
        
        all_loaded = True
        for name, filename in models_to_test:
            file_path = os.path.join(model_dir, filename)
            try:
                model = joblib.load(file_path)
                print(f"   ✅ {name}: Loaded successfully")
            except Exception as e:
                print(f"   ❌ {name}: Failed to load - {e}")
                all_loaded = False
        
        return all_loaded
        
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        return False

def main():
    """Run comprehensive verification."""
    
    print("🎯 COMPREHENSIVE VIN PREDICTION FIX VERIFICATION")
    print("=" * 60)
    
    # Run all verification tests
    tests = [
        ("Model Files Consistency", verify_model_files),
        ("Code Consistency", verify_code_consistency), 
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ VIN prediction fix is complete and consistent")
        print("✅ File naming inconsistencies have been resolved")
        print("✅ All models load successfully")
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("⚠️ Manual intervention may be required")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
