#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Feature Extraction

Find out why feature extraction is failing for the user's VINs.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_feature_extraction():
    """Debug why feature extraction fails for specific VINs."""
    
    try:
        from train_vin_predictor import extract_vin_features, decode_year
        
        print("🔍 DEBUGGING FEATURE EXTRACTION")
        print("=" * 60)
        
        # User's VINs that failed
        test_vins = [
            "KMHSH81XBCU889564",
            "MALA751AAFM098475", 
            "KMHCT41DAEU610396"
        ]
        
        # Also test a VIN that we know works
        working_vin = "9FB45RC94HM274167"
        
        print(f"Testing VINs:")
        for vin in test_vins + [working_vin]:
            print(f"  {vin}")
        print()
        
        for vin in test_vins + [working_vin]:
            print(f"--- Testing VIN: {vin} ---")
            
            try:
                # Try the training version of feature extraction
                features = extract_vin_features(vin)
                
                if features:
                    print(f"✅ Training feature extraction SUCCESS")
                    print(f"   Features: {features}")
                    
                    # Test year decoding
                    year_decoded = decode_year(features['year_code'])
                    print(f"   Year decoded: {year_decoded}")
                    
                else:
                    print(f"❌ Training feature extraction FAILED")
                    print(f"   Returned: {features}")
                
                # Also try the production version
                try:
                    from train_vin_predictor import extract_vin_features_production
                    features_prod = extract_vin_features_production(vin)
                    
                    if features_prod:
                        print(f"✅ Production feature extraction SUCCESS")
                        print(f"   Features: {features_prod}")
                    else:
                        print(f"❌ Production feature extraction FAILED")
                        print(f"   Returned: {features_prod}")
                        
                except Exception as e:
                    print(f"❌ Production feature extraction ERROR: {e}")
                
            except Exception as e:
                print(f"❌ Feature extraction ERROR: {e}")
                import traceback
                traceback.print_exc()
            
            print()
        
        # Let's also manually check what the feature extraction function expects
        print("🔍 MANUAL VIN ANALYSIS")
        print("=" * 60)
        
        for vin in test_vins[:1]:  # Just check the first one in detail
            print(f"Manual analysis of: {vin}")
            print(f"Length: {len(vin)}")
            print(f"Characters: {list(vin)}")
            
            if len(vin) >= 17:
                print(f"WMI (1-3): {vin[0:3]}")
                print(f"VDS (4-9): {vin[3:9]}")
                print(f"Year code (10): {vin[9]}")
                print(f"Plant code (11): {vin[10]}")
                print(f"Serial (12-17): {vin[11:17]}")
                
                # Check year code specifically
                year_code = vin[9]
                year_decoded = decode_year(year_code)
                print(f"Year code '{year_code}' decodes to: {year_decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error debugging feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_feature_extraction()
