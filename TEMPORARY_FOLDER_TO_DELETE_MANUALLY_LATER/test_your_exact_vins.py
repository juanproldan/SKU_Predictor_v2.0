#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Your Exact VINs from Database Screenshot

Test the exact VINs that the user provided from their database screenshot
to understand why they return "Unknown".

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_your_exact_vins():
    """Test the exact VINs from the user's database screenshot."""
    
    try:
        from unified_consolidado_processor import get_base_path
        from train_vin_predictor import extract_vin_features_production
        import joblib
        import pandas as pd
        
        print("🧪 TESTING YOUR EXACT VINs FROM DATABASE SCREENSHOT")
        print("=" * 70)
        
        # Your exact VINs from the screenshot
        your_vins = [
            "KMHSH81XBCU889564",
            "MALA751AAFM098475", 
            "KMHCT41DAEU610396"
        ]
        
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        model_dir = os.path.join(base_path, "models")
        
        print(f"📂 Database: {db_path}")
        print(f"📂 Models: {model_dir}")
        
        # First, verify these VINs exist in the database
        print(f"\n🔍 STEP 1: VERIFYING VINs EXIST IN DATABASE")
        print("=" * 70)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for vin in your_vins:
            cursor.execute("""
                SELECT vin_number, maker, model, series
                FROM processed_consolidado 
                WHERE vin_number = ?
            """, (vin,))
            
            result = cursor.fetchone()
            if result:
                vin_db, maker, model, series = result
                print(f"✅ {vin}")
                print(f"   Database: {maker} {model} {series}")
            else:
                print(f"❌ {vin} - NOT FOUND IN DATABASE!")
        
        # Load models
        print(f"\n🔍 STEP 2: LOADING MODELS")
        print("=" * 70)
        
        try:
            model_maker = joblib.load(os.path.join(model_dir, 'maker_model.joblib'))
            encoder_x_maker = joblib.load(os.path.join(model_dir, 'maker_encoder_x.joblib'))
            encoder_y_maker = joblib.load(os.path.join(model_dir, 'maker_encoder_y.joblib'))
            
            model_model = joblib.load(os.path.join(model_dir, 'model_model.joblib'))
            encoder_x_model = joblib.load(os.path.join(model_dir, 'model_encoder_x.joblib'))
            encoder_y_model = joblib.load(os.path.join(model_dir, 'model_encoder_y.joblib'))
            
            model_series = joblib.load(os.path.join(model_dir, 'series_model.joblib'))
            encoder_x_series = joblib.load(os.path.join(model_dir, 'series_encoder_x.joblib'))
            encoder_y_series = joblib.load(os.path.join(model_dir, 'series_encoder_y.joblib'))
            
            print("✅ All models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
        
        # Test predictions
        print(f"\n🔍 STEP 3: TESTING VIN PREDICTIONS")
        print("=" * 70)
        
        for i, vin in enumerate(your_vins, 1):
            print(f"\n--- Test {i}: {vin} ---")
            
            # Get expected values from database
            cursor.execute("""
                SELECT maker, model, series
                FROM processed_consolidado 
                WHERE vin_number = ?
            """, (vin,))
            
            db_result = cursor.fetchone()
            if db_result:
                expected_maker, expected_model, expected_series = db_result
                print(f"Expected: {expected_maker} {expected_model} {expected_series}")
            else:
                print("❌ VIN not found in database - skipping")
                continue
            
            # Extract features
            try:
                features = extract_vin_features_production(vin)
                if not features:
                    print("❌ Feature extraction failed")
                    continue
                
                print(f"✅ Features: {features}")
                
                # Test each prediction step
                print(f"\n   🔍 MAKER PREDICTION:")
                try:
                    X_maker = pd.DataFrame([[features['wmi']]], columns=['wmi'])
                    print(f"      Input: WMI = {features['wmi']}")
                    
                    # Check if WMI is known to the encoder
                    try:
                        X_maker_encoded = encoder_x_maker.transform(X_maker)
                        print(f"      ✅ WMI encoding successful")
                        
                        maker_pred_encoded = model_maker.predict(X_maker_encoded)
                        predicted_maker = encoder_y_maker.inverse_transform(maker_pred_encoded.reshape(-1, 1))[0][0]
                        print(f"      ✅ Predicted Maker: {predicted_maker}")
                        
                        maker_match = predicted_maker.upper() == expected_maker.upper()
                        print(f"      Match: {'✅' if maker_match else '❌'}")
                        
                    except Exception as e:
                        print(f"      ❌ WMI encoding failed: {e}")
                        print(f"      This means WMI '{features['wmi']}' was not seen during training")
                
                except Exception as e:
                    print(f"   ❌ Maker prediction error: {e}")
                
                print(f"\n   🔍 YEAR PREDICTION:")
                try:
                    X_model = pd.DataFrame([[features['year_code']]], columns=['year_code'])
                    print(f"      Input: year_code = {features['year_code']}")
                    
                    try:
                        X_model_encoded = encoder_x_model.transform(X_model)
                        print(f"      ✅ Year code encoding successful")
                        
                        model_pred_encoded = model_model.predict(X_model_encoded)
                        predicted_model = encoder_y_model.inverse_transform(model_pred_encoded.reshape(-1, 1))[0][0]
                        print(f"      ✅ Predicted Year: {predicted_model}")
                        
                        year_match = str(predicted_model) == str(expected_model)
                        print(f"      Match: {'✅' if year_match else '❌'}")
                        
                    except Exception as e:
                        print(f"      ❌ Year code encoding failed: {e}")
                        print(f"      This means year_code '{features['year_code']}' was not seen during training")
                
                except Exception as e:
                    print(f"   ❌ Year prediction error: {e}")
                
                print(f"\n   🔍 SERIES PREDICTION:")
                try:
                    X_series = pd.DataFrame([[features['wmi'], features['vds_full']]], columns=['wmi', 'vds_full'])
                    print(f"      Input: WMI = {features['wmi']}, VDS_FULL = {features['vds_full']}")
                    
                    try:
                        X_series_encoded = encoder_x_series.transform(X_series)
                        print(f"      ✅ Series encoding successful")
                        
                        series_pred_encoded = model_series.predict(X_series_encoded)
                        predicted_series = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0][0]
                        print(f"      ✅ Predicted Series: {predicted_series}")
                        
                        series_match = predicted_series.upper() == expected_series.upper()
                        print(f"      Match: {'✅' if series_match else '❌'}")
                        
                    except Exception as e:
                        print(f"      ❌ Series encoding failed: {e}")
                        print(f"      This means WMI+VDS combination was not seen during training")
                
                except Exception as e:
                    print(f"   ❌ Series prediction error: {e}")
                
            except Exception as e:
                print(f"❌ Feature extraction error: {e}")
        
        conn.close()
        
        print(f"\n" + "=" * 70)
        print("💡 ANALYSIS SUMMARY")
        print("=" * 70)
        print("If you see 'encoding failed' errors, it means:")
        print("1. The VIN patterns exist in your database")
        print("2. But they were NOT included in the training data")
        print("3. This suggests a filtering issue during training")
        print("\nNext step: Check why these VINs were excluded from training")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing VINs: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_your_exact_vins()
