#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug VIN Extraction Issues

This tool debugs VIN feature extraction failures to identify and fix issues
with the VIN cleaning and feature extraction functions.
"""

import os
import sys
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_vin_extraction():
    """Debug VIN extraction issues."""
    
    try:
        from train_vin_predictor import (
            clean_vin_for_production, 
            clean_vin_for_training,
            extract_vin_features_production,
            decode_year
        )
        
        print("🔧 VIN Extraction Debug Tool")
        print("=" * 50)
        
        # Test VINs from user examples
        test_vins = [
            "MALAT41CAJM280395",  # Hyundai - not in DB
            "MM7UR4DF7GW498254",  # Mazda - works
            "TMAJUB1E5C7T6290",   # Toyota - fails extraction
            "TMAJUB1E5C7T35032",  # Toyota - from DB screenshot
            "TMAJUB1E5C7T35539",  # Toyota - from DB screenshot
        ]
        
        for vin in test_vins:
            print(f"\n--- Debugging VIN: {vin} ---")
            print(f"📏 Length: {len(vin)}")
            print(f"🔤 Characters: {vin}")
            
            # Test basic validation
            if len(vin) != 17:
                print(f"❌ Invalid length: {len(vin)} (should be 17)")
                continue
            
            # Test character validation
            if not re.match("^[A-HJ-NPR-Z0-9]{17}$", vin):
                print(f"❌ Invalid characters detected")
                invalid_chars = [c for c in vin if not re.match("[A-HJ-NPR-Z0-9]", c)]
                print(f"   Invalid chars: {invalid_chars}")
                continue
            else:
                print(f"✅ Character validation passed")
            
            # Test production cleaning
            try:
                cleaned_prod = clean_vin_for_production(vin)
                if cleaned_prod:
                    print(f"✅ Production cleaning: {cleaned_prod}")
                else:
                    print(f"❌ Production cleaning failed")
                    continue
            except Exception as e:
                print(f"❌ Production cleaning error: {e}")
                continue
            
            # Test training cleaning
            try:
                cleaned_train = clean_vin_for_training(vin)
                if cleaned_train:
                    print(f"✅ Training cleaning: {cleaned_train}")
                else:
                    print(f"❌ Training cleaning failed")
            except Exception as e:
                print(f"❌ Training cleaning error: {e}")
            
            # Test feature extraction
            try:
                features = extract_vin_features_production(vin)
                if features:
                    print(f"✅ Feature extraction successful:")
                    for key, value in features.items():
                        print(f"   {key}: {value}")
                else:
                    print(f"❌ Feature extraction returned None")
            except Exception as e:
                print(f"❌ Feature extraction error: {e}")
                import traceback
                traceback.print_exc()
            
            # Test individual components
            if len(vin) == 17:
                print(f"\n🔍 VIN Components:")
                print(f"   WMI (1-3): {vin[0:3]}")
                print(f"   VDS (4-8): {vin[3:8]}")
                print(f"   Check (9): {vin[8]}")
                print(f"   Year (10): {vin[9]}")
                print(f"   Plant (11): {vin[10]}")
                print(f"   Serial (12-17): {vin[11:17]}")
                
                # Test year decoding
                try:
                    year_code = vin[9]
                    decoded_year = decode_year(year_code)
                    print(f"   Year decode: {year_code} → {decoded_year}")
                except Exception as e:
                    print(f"   Year decode error: {e}")
        
        # Test year decoding specifically
        print(f"\n🗓️ Testing Year Code Decoding:")
        year_codes = ['G', 'J', 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
        
        for code in year_codes[:10]:  # Test first 10
            try:
                year = decode_year(code)
                print(f"   {code} → {year}")
            except Exception as e:
                print(f"   {code} → Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_vin_extraction()
    
    if success:
        print("\n🎉 VIN debugging completed!")
    else:
        print("\n💥 VIN debugging failed!")
