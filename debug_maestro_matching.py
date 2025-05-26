#!/usr/bin/env python3
"""
Debug script to test Maestro matching with the refined approach.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_maestro_data():
    """Debug the Maestro data structure to understand the columns."""
    print("=== Debugging Maestro Data Structure ===")
    
    try:
        # Load the maestro data the same way the main app does
        import pandas as pd
        
        maestro_file_path = "data/Maestro_Equivalencias_Fixacar.xlsx"
        if os.path.exists(maestro_file_path):
            maestro_df = pd.read_excel(maestro_file_path)
            print(f"✅ Maestro file loaded: {len(maestro_df)} rows")
            print(f"✅ Columns: {list(maestro_df.columns)}")
            
            # Show sample data
            print(f"\n📋 Sample Maestro entries:")
            for i, row in maestro_df.head(3).iterrows():
                print(f"  Row {i}:")
                print(f"    VIN_Make: '{row.get('VIN_Make', 'N/A')}'")
                print(f"    VIN_Year_Min: '{row.get('VIN_Year_Min', 'N/A')}'")
                print(f"    VIN_Series_Trim: '{row.get('VIN_Series_Trim', 'N/A')}'")
                print(f"    Normalized_Description_Input: '{row.get('Normalized_Description_Input', 'N/A')}'")
                print(f"    Confirmed_SKU: '{row.get('Confirmed_SKU', 'N/A')}'")
                print()
            
            return maestro_df.to_dict('records')
        else:
            print(f"❌ Maestro file not found: {maestro_file_path}")
            return []
            
    except Exception as e:
        print(f"❌ Error loading Maestro data: {e}")
        return []

def test_maestro_matching():
    """Test the Maestro matching logic with real data."""
    print("\n=== Testing Maestro Matching Logic ===")
    
    try:
        from core.prediction import get_standardized_predictor
        from utils.text_utils import get_synonym_expander
        
        # Load real maestro data
        maestro_data = debug_maestro_data()
        if not maestro_data:
            print("❌ No Maestro data available for testing")
            return False
        
        # Initialize predictor with real data
        predictor = get_standardized_predictor()
        
        # Load synonym expander
        expander = get_synonym_expander()
        
        try:
            predictor.initialize(
                maestro_data=maestro_data,
                db_path="data/historical_parts.db",  # May not exist
                neural_network_predictor=None  # Not needed for this test
            )
            print(f"✅ Predictor initialized with {len(maestro_data)} maestro entries")
        except Exception as e:
            print(f"⚠️  Predictor initialization had issues: {e}")
        
        # Test with the example from the screenshot
        test_cases = [
            {
                'make': 'RENAULT',
                'model_year': '2018',
                'series': 'LOGAN II/LIFE',
                'part_description': 'capo'
            },
            {
                'make': 'RENAULT',
                'model_year': '2018',
                'series': 'LOGAN II/LIFE',
                'part_description': 'guardafango der'
            },
            {
                'make': 'RENAULT',
                'model_year': '2018',
                'series': 'LOGAN II/LIFE',
                'part_description': 'rejilla paragolpes del'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}: {test_case}")
            
            # Check if we can find exact matches in maestro data
            matches_found = 0
            for maestro_entry in maestro_data:
                make_match = maestro_entry.get('VIN_Make', '').upper() == test_case['make'].upper()
                year_match = str(maestro_entry.get('VIN_Year_Min', '')) == test_case['model_year']
                series_match = maestro_entry.get('VIN_Series_Trim', '').upper() == test_case['series'].upper()
                desc_match = maestro_entry.get('Normalized_Description_Input', '').lower() == test_case['part_description'].lower()
                
                if make_match and year_match and series_match:
                    matches_found += 1
                    if desc_match:
                        print(f"  ✅ EXACT MATCH found!")
                        print(f"    SKU: {maestro_entry.get('Confirmed_SKU')}")
                        print(f"    Maestro Entry: {maestro_entry}")
                        break
                    else:
                        # Show what descriptions are available for this make/year/series
                        available_desc = maestro_entry.get('Normalized_Description_Input', '')
                        if available_desc:
                            print(f"  📝 Available description: '{available_desc}'")
            
            if matches_found == 0:
                print(f"  ❌ No matches found for Make/Year/Series combination")
                # Show what's available
                print(f"  🔍 Looking for similar entries...")
                for maestro_entry in maestro_data[:10]:  # Check first 10
                    if maestro_entry.get('VIN_Make', '').upper() == test_case['make'].upper():
                        print(f"    Found {test_case['make']} entry: Year={maestro_entry.get('VIN_Year_Min')}, Series='{maestro_entry.get('VIN_Series_Trim')}'")
            else:
                print(f"  📊 Found {matches_found} entries for Make/Year/Series combination")
            
            # Test with predictor
            try:
                results = predictor.predict(**test_case)
                maestro_results = results.get('maestro_lookup', [])
                print(f"  🤖 Predictor found {len(maestro_results)} Maestro results")
                for result in maestro_results:
                    print(f"    - SKU: {result.sku}, Confidence: {result.confidence}")
            except Exception as e:
                print(f"  ❌ Predictor failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Maestro matching test failed: {e}")
        return False

def main():
    """Run Maestro debugging tests."""
    print("🔧 Debugging Maestro Matching Logic")
    print("=" * 50)
    
    tests = [
        ("Maestro Data Structure", debug_maestro_data),
        ("Maestro Matching Logic", test_maestro_matching)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} completed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print("🎯 Debug session complete!")

if __name__ == "__main__":
    main()
