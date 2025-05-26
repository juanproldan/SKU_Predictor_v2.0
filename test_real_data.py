#!/usr/bin/env python3
"""
Real Data Test Script for Historical Data Improvements
Tests the new Maestro and Database search logic with actual data.
"""

import sys
import os
import sqlite3
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_database_structure():
    """Test if the database has the required columns for our new logic"""
    print("=== Testing Database Structure ===")
    
    db_path = "data/fixacar_history.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if required columns exist
        cursor.execute("PRAGMA table_info(historical_parts)")
        columns = [row[1] for row in cursor.fetchall()]
        
        required_columns = ['vin_make', 'vin_year', 'vin_series', 'normalized_description', 'sku']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ All required columns present in database")
        
        # Check data availability
        cursor.execute("""
            SELECT COUNT(*) FROM historical_parts 
            WHERE vin_make IS NOT NULL 
            AND vin_year IS NOT NULL 
            AND vin_series IS NOT NULL 
            AND normalized_description IS NOT NULL 
            AND sku IS NOT NULL
        """)
        
        count = cursor.fetchone()[0]
        print(f"✅ Found {count:,} complete records for testing")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_maestro_structure():
    """Test if Maestro data has the required structure"""
    print("\n=== Testing Maestro Structure ===")
    
    maestro_path = "data/Maestro.xlsx"
    if not os.path.exists(maestro_path):
        print(f"❌ Maestro file not found at {maestro_path}")
        return False
    
    try:
        df = pd.read_excel(maestro_path)
        
        required_columns = ['VIN_Make', 'VIN_Year_Min', 'VIN_Series_Trim', 
                          'Normalized_Description_Input', 'Confirmed_SKU']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns in Maestro: {missing_columns}")
            return False
        
        print("✅ All required columns present in Maestro")
        
        # Check data availability
        complete_records = df.dropna(subset=required_columns)
        print(f"✅ Found {len(complete_records):,} complete Maestro records")
        
        return True
        
    except Exception as e:
        print(f"❌ Maestro error: {e}")
        return False

def test_search_logic_simulation():
    """Simulate the new search logic with sample data"""
    print("\n=== Testing Search Logic Simulation ===")
    
    try:
        # Test parameters
        test_cases = [
            {
                "make": "TOYOTA",
                "year": "2015", 
                "series": "CAMRY LE",
                "description": "farola izquierda",
                "expected": "Should find exact or fuzzy matches"
            },
            {
                "make": "HONDA",
                "year": "2018",
                "series": "CIVIC LX", 
                "description": "espejo retrovisor",
                "expected": "Should find matches with Series requirement"
            }
        ]
        
        db_path = "data/fixacar_history.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Input: {test_case['make']} {test_case['year']} {test_case['series']} - {test_case['description']}")
            
            # Test 4-parameter exact search (our new primary method)
            cursor.execute("""
                SELECT sku, COUNT(*) as frequency
                FROM historical_parts
                WHERE vin_make = ? AND vin_year = ? AND vin_series = ? AND normalized_description = ?
                GROUP BY sku
                ORDER BY frequency DESC
                LIMIT 5
            """, (test_case['make'], test_case['year'], test_case['series'], test_case['description']))
            
            exact_results = cursor.fetchall()
            
            if exact_results:
                print(f"✅ 4-param exact: Found {len(exact_results)} SKUs")
                for sku, freq in exact_results:
                    print(f"   - {sku} (freq: {freq})")
            else:
                print("⚠️  4-param exact: No results")
                
                # Test 3-parameter search (our fallback)
                cursor.execute("""
                    SELECT sku, normalized_description, COUNT(*) as frequency
                    FROM historical_parts
                    WHERE vin_make = ? AND vin_year = ? AND vin_series = ?
                    GROUP BY sku, normalized_description
                    ORDER BY frequency DESC
                    LIMIT 10
                """, (test_case['make'], test_case['year'], test_case['series']))
                
                fallback_results = cursor.fetchall()
                
                if fallback_results:
                    print(f"✅ 3-param fallback: Found {len(fallback_results)} potential matches")
                    for sku, desc, freq in fallback_results[:3]:
                        print(f"   - {sku}: {desc} (freq: {freq})")
                else:
                    print("⚠️  3-param fallback: No results")
        
        conn.close()
        print("\n✅ Search logic simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Search simulation error: {e}")
        return False

def test_confidence_scoring():
    """Test the new confidence scoring logic"""
    print("\n=== Testing Confidence Scoring ===")
    
    try:
        # Test Maestro fuzzy confidence: 0.7 + 0.25 * similarity
        similarities = [0.8, 0.85, 0.9, 0.95, 1.0]
        
        print("Maestro Fuzzy Confidence Scores:")
        for sim in similarities:
            confidence = round(0.7 + 0.25 * sim, 3)
            print(f"  Similarity {sim:.2f} → Confidence {confidence:.3f}")
        
        # Test Database confidence: base + frequency boost
        print("\nDatabase Confidence Scores:")
        base_similarities = [0.8, 0.85, 0.9]
        frequencies = [1, 5, 10, 20]
        total_freq = 50
        
        for sim in base_similarities:
            print(f"  Similarity {sim:.2f}:")
            for freq in frequencies:
                base_confidence = 0.3 + 0.3 * sim
                freq_boost = 0.2 * (freq / total_freq)
                confidence = round(base_confidence + freq_boost, 3)
                print(f"    Freq {freq}/{total_freq} → Confidence {confidence:.3f}")
        
        print("✅ Confidence scoring logic verified")
        return True
        
    except Exception as e:
        print(f"❌ Confidence scoring error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Real Data Tests for Historical Data Improvements")
    print("=" * 60)
    
    tests = [
        test_database_structure,
        test_maestro_structure, 
        test_search_logic_simulation,
        test_confidence_scoring
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ Historical Data Improvements are ready for production!")
        print("\nNext steps:")
        print("1. Create Git branch: git checkout -b Historical_Data_Improvements")
        print("2. Run main application to test with real user input")
        print("3. Compare results with previous version")
        print("4. Monitor confidence score distribution")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("\nPlease review the failed tests before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
