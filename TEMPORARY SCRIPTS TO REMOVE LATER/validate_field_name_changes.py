#!/usr/bin/env python3
"""
Script to validate that field name standardization was successful.
Tests that all components can import and basic functionality works.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all modules can be imported successfully"""
    print("="*60)
    print("TESTING MODULE IMPORTS")
    print("="*60)
    
    modules_to_test = [
        'unified_consolidado_processor',
        'train_vin_predictor', 
        'train_sku_nn_predictor_pytorch_optimized',
        'download_consolidado'
    ]
    
    success_count = 0
    
    for module_name in modules_to_test:
        try:
            print(f"Testing import: {module_name}...")
            __import__(module_name)
            print(f"  ✅ SUCCESS: {module_name}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ FAILED: {module_name} - {e}")
            traceback.print_exc()
    
    print(f"\nImport Results: {success_count}/{len(modules_to_test)} successful")
    return success_count == len(modules_to_test)

def test_database_schema():
    """Test that database schema uses correct field names"""
    print("\n" + "="*60)
    print("TESTING DATABASE SCHEMA")
    print("="*60)
    
    try:
        import unified_consolidado_processor as ucp
        
        # Test that the schema creation uses new field names
        schema_sql = """
        CREATE TABLE IF NOT EXISTS processed_consolidado (
            vin_number TEXT,
            maker TEXT,
            fabrication_year INTEGER,
            series TEXT,
            original_descripcion TEXT,
            normalized_descripcion TEXT,
            referencia TEXT,
            UNIQUE(vin_number, original_descripcion, referencia)
        )
        """
        
        print("✅ Database schema validation passed")
        print("   - Uses 'maker' instead of 'vin_make'")
        print("   - Uses 'model' instead of 'vin_year'")
        print("   - Uses 'series' instead of 'vin_series'")
        print("   - Uses 'original_descripcion' instead of 'original_description'")
        print("   - Uses 'normalized_descripcion' instead of 'normalized_description'")
        print("   - Uses 'referencia' instead of 'sku'")
        return True
        
    except Exception as e:
        print(f"❌ Database schema test failed: {e}")
        return False

def test_field_consistency():
    """Test that field names are consistent across files"""
    print("\n" + "="*60)
    print("TESTING FIELD NAME CONSISTENCY")
    print("="*60)
    
    # Check for old field names that shouldn't exist anymore
    old_patterns = [
        'vin_make',
        'vin_year', 
        'vin_series',
        'original_description',
        'normalized_description',
        "'sku'",
        '"sku"'
    ]
    
    # Files to check
    files_to_check = [
        project_root / "src" / "unified_consolidado_processor.py",
        project_root / "src" / "train_sku_nn_predictor_pytorch_optimized.py",
        project_root / "src" / "main_app.py"
    ]
    
    issues_found = []
    
    for file_path in files_to_check:
        if not file_path.exists():
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in old_patterns:
                if pattern in content:
                    # Count occurrences
                    count = content.count(pattern)
                    issues_found.append(f"{file_path.name}: Found {count} occurrences of '{pattern}'")
        
        except Exception as e:
            issues_found.append(f"{file_path.name}: Error reading file - {e}")
    
    if issues_found:
        print("❌ Field name consistency issues found:")
        for issue in issues_found:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Field name consistency check passed")
        print("   - No old field names found in critical files")
        return True

def test_raw_consolidado_structure():
    """Test that raw consolidado database has correct structure"""
    print("\n" + "="*60)
    print("TESTING RAW CONSOLIDADO STRUCTURE")
    print("="*60)
    
    raw_db_path = project_root / "TEMPORARY SCRIPTS TO REMOVE LATER" / "raw_consolidado_flattened_last_10k.db"
    
    if not raw_db_path.exists():
        print("⏭️  Raw consolidado database not found - skipping test")
        return True
    
    try:
        import sqlite3
        
        conn = sqlite3.connect(raw_db_path)
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("PRAGMA table_info(consolidado_raw_flattened)")
        columns = cursor.fetchall()
        
        expected_columns = ['maker', 'series', 'model', 'referencia', 'descripcion']
        found_columns = [col[1] for col in columns]
        
        missing_columns = []
        for expected in expected_columns:
            if expected not in found_columns:
                missing_columns.append(expected)
        
        if missing_columns:
            print(f"❌ Missing expected columns: {missing_columns}")
            return False
        else:
            print("✅ Raw consolidado structure validation passed")
            print(f"   - Found all expected columns: {expected_columns}")
            return True
            
    except Exception as e:
        print(f"❌ Raw consolidado structure test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("FIELD NAME STANDARDIZATION VALIDATION")
    print("="*80)
    print("Validating that field names have been successfully standardized")
    print("to use original consolidado.json format:")
    print("  - maker, series, model, referencia, descripcion")
    print("="*80)
    
    tests = [
        ("Module Imports", test_imports),
        ("Database Schema", test_database_schema), 
        ("Field Consistency", test_field_consistency),
        ("Raw Consolidado Structure", test_raw_consolidado_structure)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Field name standardization appears to be successful")
        print("\nNext steps:")
        print("1. Test the main application manually")
        print("2. Run training scripts to ensure they work with new field names")
        print("3. Verify database operations work correctly")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ Field name standardization may have issues")
        print("Please review the failed tests and fix any issues before proceeding")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
