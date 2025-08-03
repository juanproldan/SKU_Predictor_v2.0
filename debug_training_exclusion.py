#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Training Exclusion

Find out exactly why the user's VINs were excluded from training
by simulating the exact training process.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import sqlite3
import pandas as pd
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_training_exclusion():
    """Debug why specific VINs were excluded from training."""
    
    try:
        from unified_consolidado_processor import get_base_path
        from train_vin_predictor import extract_vin_features, decode_year
        
        print("🔍 DEBUGGING TRAINING EXCLUSION")
        print("=" * 60)
        
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        
        # User's VINs
        test_vins = [
            "KMHSH81XBCU889564",
            "MALA751AAFM098475", 
            "KMHCT41DAEU610396"
        ]
        
        conn = sqlite3.connect(db_path)
        
        # Run the exact same query as training
        print("🔍 STEP 1: RUNNING TRAINING QUERY")
        query = """
        SELECT DISTINCT vin_number, maker as maker, model as model, series as series
        FROM processed_consolidado
        WHERE vin_number IS NOT NULL
          AND maker IS NOT NULL
          AND model IS NOT NULL
          AND series IS NOT NULL
          AND LENGTH(vin_number) = 17
          AND vin_number != '00000000000000000'
          AND vin_number NOT LIKE '%00000000000000000%'
        """
        
        df_raw = pd.read_sql_query(query, conn)
        print(f"Total records from training query: {len(df_raw)}")
        
        # Check if our test VINs are in the raw query results
        print(f"\n🔍 STEP 2: CHECKING IF TEST VINs ARE IN QUERY RESULTS")
        for vin in test_vins:
            vin_in_query = df_raw[df_raw['vin_number'] == vin]
            if len(vin_in_query) > 0:
                row = vin_in_query.iloc[0]
                print(f"✅ {vin} - Found in query results")
                print(f"   Maker: {row['maker']}, Model: {row['model']}, Series: {row['series']}")
            else:
                print(f"❌ {vin} - NOT found in query results")
        
        # Now simulate the filtering process
        print(f"\n🔍 STEP 3: SIMULATING TRAINING FILTERING")
        
        all_data = []
        valid_vins = 0
        invalid_vins = 0
        filtering_stats = {
            'empty_null': 0,
            'wrong_length': 0,
            'invalid_chars': 0,
            'ioq_chars': 0,
            'suspicious_patterns': 0,
            'invalid_check_digit_pos': 0,
            'invalid_year_code': 0,
            'feature_extraction_failed': 0
        }
        
        test_vin_results = {}
        
        for _, row in df_raw.iterrows():
            vin = row['vin_number']
            is_test_vin = vin in test_vins
            
            if is_test_vin:
                print(f"\n--- Processing test VIN: {vin} ---")
            
            # Track specific filtering reasons
            if not vin:
                filtering_stats['empty_null'] += 1
                invalid_vins += 1
                if is_test_vin:
                    test_vin_results[vin] = "FILTERED: Empty/null VIN"
                continue

            vin_str = str(vin).upper().strip()
            if len(vin_str) != 17:
                filtering_stats['wrong_length'] += 1
                invalid_vins += 1
                if is_test_vin:
                    test_vin_results[vin] = f"FILTERED: Wrong length ({len(vin_str)})"
                continue

            if not re.match("^[A-HJ-NPR-Z0-9]{17}$", vin_str):
                filtering_stats['invalid_chars'] += 1
                invalid_vins += 1
                if is_test_vin:
                    test_vin_results[vin] = "FILTERED: Invalid characters"
                continue

            features = extract_vin_features(vin)
            if not features:
                filtering_stats['feature_extraction_failed'] += 1
                invalid_vins += 1
                if is_test_vin:
                    test_vin_results[vin] = "FILTERED: Feature extraction failed"
                continue
            
            if is_test_vin:
                print(f"   ✅ Passed all filters!")
                print(f"   Features: {features}")
                test_vin_results[vin] = "PASSED ALL FILTERS"
            
            valid_vins += 1
            year_target = row['model']
            year_decoded = decode_year(features['year_code'])
            
            all_data.append({
                'vin': vin,
                'wmi': features['wmi'],
                'vds': features['vds'],
                'year_code': features['year_code'],
                'plant_code': features['plant_code'],
                'vds_full': features['vds_full'],
                'maker': row['maker'],
                'model': year_target,
                'series': row['series'],
                'year_decoded': year_decoded
            })
        
        conn.close()
        
        print(f"\n🔍 STEP 4: FILTERING RESULTS")
        print("=" * 60)
        print(f"Total VINs processed: {len(df_raw)}")
        print(f"Valid VINs: {valid_vins}")
        print(f"Invalid VINs: {invalid_vins}")
        print(f"Filtering stats: {filtering_stats}")
        
        print(f"\n🎯 TEST VIN RESULTS:")
        for vin, result in test_vin_results.items():
            print(f"   {vin}: {result}")
        
        # Check if the issue is in the frequency filtering
        if all_data:
            df_processed = pd.DataFrame(all_data)
            
            print(f"\n🔍 STEP 5: CHECKING FREQUENCY FILTERING")
            
            # Check WMI frequencies
            wmi_counts = df_processed['wmi'].value_counts()
            print(f"WMI frequency analysis:")
            
            test_wmis = ['KMH', 'MAL']
            for wmi in test_wmis:
                if wmi in wmi_counts:
                    count = wmi_counts[wmi]
                    print(f"   {wmi}: {count} occurrences")
                    if count < 10:  # Assuming minimum threshold
                        print(f"      ⚠️ Low frequency - might be filtered out")
                else:
                    print(f"   {wmi}: 0 occurrences - NOT IN PROCESSED DATA")
        
        return True
        
    except Exception as e:
        print(f"❌ Error debugging training exclusion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_training_exclusion()
