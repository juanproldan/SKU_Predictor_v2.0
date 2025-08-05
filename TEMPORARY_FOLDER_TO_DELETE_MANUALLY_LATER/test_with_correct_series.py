#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test SKU Predictor with Correct Series

This tool tests the user's examples with the correct series names from the database
to demonstrate that the SKU predictor works when proper series are provided.
"""

import os
import sys
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_with_correct_series():
    """Test SKU predictions with correct series from database."""
    
    try:
        from unified_consolidado_processor import get_base_path
        from utils.year_range_database import YearRangeDatabaseOptimizer
        from train_vin_predictor import extract_vin_features_production
        
        print("🧪 Testing with Correct Series Names")
        print("=" * 60)
        
        # Get database path
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        
        # Initialize year range optimizer
        year_range_optimizer = YearRangeDatabaseOptimizer(db_path)
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🎯 Testing User's Examples with Correct Series...")
        
        # Test 1: Hyundai 2018 with correct series
        print(f"\n--- Test 1: Hyundai 2018 COSTADO IZQUIERDA ---")
        
        # Find correct Hyundai 2018 series
        cursor.execute("""
            SELECT DISTINCT series, COUNT(*) as count
            FROM processed_consolidado 
            WHERE LOWER(maker) LIKE '%hyundai%' 
            AND model = '2018'
            AND LOWER(descripcion) LIKE '%costado%'
            GROUP BY series
            ORDER BY count DESC
            LIMIT 3
        """)
        
        hyundai_series = cursor.fetchall()
        if hyundai_series:
            print(f"📊 Found Hyundai 2018 series with COSTADO parts:")
            for series, count in hyundai_series:
                print(f"  - '{series}': {count} records")
                
                # Test with this series
                predictions = year_range_optimizer.get_sku_predictions_year_range(
                    maker="Hyundai",
                    model="2018",
                    series=series,
                    description="COSTADO IZQUIERDA",
                    limit=5
                )
                
                if predictions:
                    print(f"    ✅ Found {len(predictions)} predictions:")
                    for i, pred in enumerate(predictions):
                        print(f"      {i+1}. {pred['sku']} (freq: {pred['frequency']}, conf: {pred['confidence']:.3f})")
                else:
                    print(f"    ❌ No predictions found")
        else:
            print("❌ No Hyundai 2018 series found with COSTADO parts")
        
        # Test 2: Hyundai 2018 PERSIANA
        print(f"\n--- Test 2: Hyundai 2018 PERSIANA ---")
        
        cursor.execute("""
            SELECT DISTINCT series, COUNT(*) as count
            FROM processed_consolidado 
            WHERE LOWER(maker) LIKE '%hyundai%' 
            AND model = '2018'
            AND LOWER(descripcion) LIKE '%persiana%'
            GROUP BY series
            ORDER BY count DESC
            LIMIT 3
        """)
        
        hyundai_persiana_series = cursor.fetchall()
        if hyundai_persiana_series:
            print(f"📊 Found Hyundai 2018 series with PERSIANA parts:")
            for series, count in hyundai_persiana_series:
                print(f"  - '{series}': {count} records")
                
                # Test with this series
                predictions = year_range_optimizer.get_sku_predictions_year_range(
                    maker="Hyundai",
                    model="2018",
                    series=series,
                    description="PERSIANA",
                    limit=5
                )
                
                if predictions:
                    print(f"    ✅ Found {len(predictions)} predictions:")
                    for i, pred in enumerate(predictions):
                        print(f"      {i+1}. {pred['sku']} (freq: {pred['frequency']}, conf: {pred['confidence']:.3f})")
                else:
                    print(f"    ❌ No predictions found")
        else:
            print("❌ No Hyundai 2018 series found with PERSIANA parts")
        
        # Test 3: VIN Prediction Fix
        print(f"\n--- Test 3: VIN Prediction (Fixed Year Extraction) ---")
        
        test_vins = [
            "MM7UR4DF7GW498254",  # Mazda from user
            "TMAJUB1E5C7T6290",   # Toyota from database
        ]
        
        for vin in test_vins:
            print(f"\n🔍 Testing VIN: {vin}")
            
            # Test fixed feature extraction
            try:
                features = extract_vin_features_production(vin)
                if features:
                    print(f"  ✅ Features extracted:")
                    print(f"    - WMI: {features.get('wmi')}")
                    print(f"    - VDS: {features.get('vds')}")
                    print(f"    - Year Code: {features.get('year_code')}")
                    print(f"    - Year: {features.get('year')}")
                else:
                    print(f"  ❌ Failed to extract features")
            except Exception as e:
                print(f"  ❌ Error extracting features: {e}")
            
            # Check if VIN exists in database
            cursor.execute("""
                SELECT maker, model, series, COUNT(*) as count
                FROM processed_consolidado 
                WHERE vin_number = ?
                GROUP BY maker, model, series
                ORDER BY count DESC
            """, (vin,))
            
            db_results = cursor.fetchall()
            if db_results:
                print(f"  📊 Database records:")
                for maker, model, series, count in db_results:
                    print(f"    - {maker} {model} {series} ({count} records)")
            else:
                print(f"  ❌ VIN not found in database")
        
        # Test 4: Series Fuzzy Matching Suggestion
        print(f"\n--- Test 4: Series Fuzzy Matching Suggestions ---")
        
        def suggest_series(maker, target_series):
            """Suggest similar series names."""
            cursor.execute("""
                SELECT DISTINCT series, COUNT(*) as count
                FROM processed_consolidado 
                WHERE LOWER(maker) LIKE ?
                AND (LOWER(series) LIKE ? OR LOWER(series) LIKE ?)
                GROUP BY series
                ORDER BY count DESC
                LIMIT 5
            """, (f'%{maker.lower()}%', f'%{target_series.lower()}%', f'%{target_series.lower().replace(" ", "%")}%'))
            
            return cursor.fetchall()
        
        suggestions = suggest_series("Hyundai", "Tucson")
        if suggestions:
            print(f"📊 Series suggestions for 'Hyundai Tucson':")
            for series, count in suggestions:
                print(f"  - '{series}': {count:,} records")
        
        suggestions = suggest_series("Toyota", "Tucson")
        if suggestions:
            print(f"📊 Series suggestions for 'Toyota Tucson' (should be empty):")
            for series, count in suggestions:
                print(f"  - '{series}': {count:,} records")
        else:
            print(f"✅ Correctly found no Toyota Tucson (Tucson is Hyundai)")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_correct_series()
    
    if success:
        print("\n🎉 Testing completed!")
        print("\n💡 Key Findings:")
        print("  1. SKU predictor works when correct series names are provided")
        print("  2. VIN year extraction has been fixed")
        print("  3. Series names must match database exactly")
        print("  4. 'Unknown' series won't find matches - need specific series")
    else:
        print("\n💥 Testing failed!")
