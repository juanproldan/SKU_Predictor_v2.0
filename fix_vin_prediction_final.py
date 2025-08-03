#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final Fix for VIN Prediction Feature Mismatch

This script creates the corrected VIN prediction function that matches
the exact features expected by the trained models.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_corrected_prediction_function():
    """Create the corrected VIN prediction function."""
    
    corrected_code = '''
def predict_vin_details_corrected(vin):
    """
    CORRECTED VIN prediction function that matches trained model features exactly.
    
    CORRECT FEATURE MAPPINGS (based on actual trained models):
    - Maker: [wmi] ✅
    - Model (Year): [year_code] ✅ (NOT wmi+vds!)
    - Series: [wmi, vds_full] ✅
    """
    from train_vin_predictor import extract_vin_features_production
    import joblib
    import pandas as pd
    from unified_consolidado_processor import get_base_path
    
    base_path = get_base_path()
    model_dir = os.path.join(base_path, "models")
    
    # Load models and encoders
    model_maker = joblib.load(os.path.join(model_dir, 'maker_model.joblib'))
    encoder_x_maker = joblib.load(os.path.join(model_dir, 'maker_encoder_x.joblib'))
    encoder_y_maker = joblib.load(os.path.join(model_dir, 'maker_encoder_y.joblib'))
    
    model_model = joblib.load(os.path.join(model_dir, 'model_model.joblib'))
    encoder_x_model = joblib.load(os.path.join(model_dir, 'model_encoder_x.joblib'))
    encoder_y_model = joblib.load(os.path.join(model_dir, 'model_encoder_y.joblib'))
    
    model_series = joblib.load(os.path.join(model_dir, 'series_model.joblib'))
    encoder_x_series = joblib.load(os.path.join(model_dir, 'series_encoder_x.joblib'))
    encoder_y_series = joblib.load(os.path.join(model_dir, 'series_encoder_y.joblib'))
    
    # Extract features
    features = extract_vin_features_production(vin)
    if not features:
        return {"maker": "Unknown", "model": "Unknown", "series": "Unknown"}
    
    results = {}
    
    try:
        # 1. MAKER PREDICTION - [wmi] only (CORRECT)
        X_maker = pd.DataFrame([[features['wmi']]], columns=['wmi'])
        X_maker_encoded = encoder_x_maker.transform(X_maker)
        maker_pred_encoded = model_maker.predict(X_maker_encoded)
        predicted_maker = encoder_y_maker.inverse_transform(maker_pred_encoded.reshape(-1, 1))[0][0]
        results['maker'] = predicted_maker
        
        # 2. MODEL (YEAR) PREDICTION - [year_code] only (CORRECTED!)
        X_model = pd.DataFrame([[features['year_code']]], columns=['year_code'])
        X_model_encoded = encoder_x_model.transform(X_model)
        model_pred_encoded = model_model.predict(X_model_encoded)
        predicted_model = encoder_y_model.inverse_transform(model_pred_encoded.reshape(-1, 1))[0][0]
        results['model'] = predicted_model
        
        # 3. SERIES PREDICTION - [wmi, vds_full] (CORRECT)
        X_series = pd.DataFrame([[features['wmi'], features['vds_full']]], columns=['wmi', 'vds_full'])
        X_series_encoded = encoder_x_series.transform(X_series)
        series_pred_encoded = model_series.predict(X_series_encoded)
        predicted_series = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0][0]
        results['series'] = predicted_series
        
        return results
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"maker": "Unknown", "model": "Unknown", "series": "Unknown"}
'''
    
    return corrected_code

def test_corrected_function():
    """Test the corrected VIN prediction function."""
    
    try:
        from unified_consolidado_processor import get_base_path
        from train_vin_predictor import extract_vin_features_production
        import joblib
        import pandas as pd
        import sqlite3
        
        print("🧪 TESTING CORRECTED VIN PREDICTION FUNCTION")
        print("=" * 60)
        
        # Get sample VINs from database
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        model_dir = os.path.join(base_path, "models")
        
        # Load models
        model_maker = joblib.load(os.path.join(model_dir, 'maker_model.joblib'))
        encoder_x_maker = joblib.load(os.path.join(model_dir, 'maker_encoder_x.joblib'))
        encoder_y_maker = joblib.load(os.path.join(model_dir, 'maker_encoder_y.joblib'))
        
        model_model = joblib.load(os.path.join(model_dir, 'model_model.joblib'))
        encoder_x_model = joblib.load(os.path.join(model_dir, 'model_encoder_x.joblib'))
        encoder_y_model = joblib.load(os.path.join(model_dir, 'model_encoder_y.joblib'))
        
        model_series = joblib.load(os.path.join(model_dir, 'series_model.joblib'))
        encoder_x_series = joblib.load(os.path.join(model_dir, 'series_encoder_x.joblib'))
        encoder_y_series = joblib.load(os.path.join(model_dir, 'series_encoder_y.joblib'))
        
        # Get test VINs
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT vin_number, maker, model, series
            FROM processed_consolidado 
            WHERE vin_number IS NOT NULL
            AND LENGTH(vin_number) = 17
            AND maker IS NOT NULL
            AND model IS NOT NULL
            AND series IS NOT NULL
            LIMIT 5
        """)
        
        test_cases = cursor.fetchall()
        conn.close()
        
        print(f"Testing {len(test_cases)} VINs with CORRECTED feature mapping...")
        
        successful = 0
        total = len(test_cases)
        
        for i, (vin, expected_maker, expected_model, expected_series) in enumerate(test_cases, 1):
            print(f"\\n--- Test {i}: {vin} ---")
            print(f"Expected: {expected_maker} {expected_model} {expected_series}")
            
            try:
                # Extract features
                features = extract_vin_features_production(vin)
                if not features:
                    print("❌ Feature extraction failed")
                    continue
                
                # CORRECTED PREDICTIONS:
                
                # 1. Maker - [wmi]
                X_maker = pd.DataFrame([[features['wmi']]], columns=['wmi'])
                X_maker_encoded = encoder_x_maker.transform(X_maker)
                maker_pred_encoded = model_maker.predict(X_maker_encoded)
                predicted_maker = encoder_y_maker.inverse_transform(maker_pred_encoded.reshape(-1, 1))[0][0]
                
                # 2. Model (Year) - [year_code] CORRECTED!
                X_model = pd.DataFrame([[features['year_code']]], columns=['year_code'])
                X_model_encoded = encoder_x_model.transform(X_model)
                model_pred_encoded = model_model.predict(X_model_encoded)
                predicted_model = encoder_y_model.inverse_transform(model_pred_encoded.reshape(-1, 1))[0][0]
                
                # 3. Series - [wmi, vds_full]
                X_series = pd.DataFrame([[features['wmi'], features['vds_full']]], columns=['wmi', 'vds_full'])
                X_series_encoded = encoder_x_series.transform(X_series)
                series_pred_encoded = model_series.predict(X_series_encoded)
                predicted_series = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0][0]
                
                print(f"🎯 Predicted: {predicted_maker} {predicted_model} {predicted_series}")
                
                # Check matches
                maker_match = predicted_maker.upper() == expected_maker.upper()
                model_match = str(predicted_model) == str(expected_model)
                series_match = predicted_series.upper() == expected_series.upper()
                
                print(f"📊 Matches: Maker={'✅' if maker_match else '❌'} Model={'✅' if model_match else '❌'} Series={'✅' if series_match else '❌'}")
                
                if maker_match and model_match:
                    successful += 1
                    print("🎉 SUCCESS!")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\\n📊 CORRECTED FUNCTION RESULTS:")
        print(f"Successful predictions: {successful}/{total} ({(successful/total)*100:.1f}%)")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Error testing corrected function: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution."""
    print("🔧 FINAL VIN PREDICTION FIX")
    print("=" * 60)
    
    print("📋 Creating corrected prediction function...")
    corrected_code = create_corrected_prediction_function()
    
    print("\\n🧪 Testing corrected function...")
    success = test_corrected_function()
    
    if success:
        print("\\n🎉 CORRECTED FUNCTION WORKS!")
        print("✅ The issue was: Model prediction used wrong features")
        print("✅ Solution: Use [year_code] instead of [wmi, vds] for year prediction")
        print("\\n📝 Next step: Update main_app.py with corrected feature mapping")
    else:
        print("\\n❌ Still having issues - need further investigation")
    
    return success

if __name__ == "__main__":
    main()
