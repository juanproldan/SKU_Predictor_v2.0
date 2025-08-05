#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Specific VIN with Corrected Feature Mapping

Test the corrected VIN prediction with a known good VIN.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_specific_vin():
    """Test with a specific VIN that we know exists."""
    
    try:
        from unified_consolidado_processor import get_base_path
        from train_vin_predictor import extract_vin_features_production
        import joblib
        import pandas as pd
        
        print("🧪 TESTING CORRECTED VIN PREDICTION WITH SPECIFIC VIN")
        print("=" * 60)
        
        base_path = get_base_path()
        model_dir = os.path.join(base_path, "models")
        
        # Load models
        print("📂 Loading models...")
        model_maker = joblib.load(os.path.join(model_dir, 'maker_model.joblib'))
        encoder_x_maker = joblib.load(os.path.join(model_dir, 'maker_encoder_x.joblib'))
        encoder_y_maker = joblib.load(os.path.join(model_dir, 'maker_encoder_y.joblib'))
        
        model_model = joblib.load(os.path.join(model_dir, 'model_model.joblib'))
        encoder_x_model = joblib.load(os.path.join(model_dir, 'model_encoder_x.joblib'))
        encoder_y_model = joblib.load(os.path.join(model_dir, 'model_encoder_y.joblib'))
        
        model_series = joblib.load(os.path.join(model_dir, 'series_model.joblib'))
        encoder_x_series = joblib.load(os.path.join(model_dir, 'series_encoder_x.joblib'))
        encoder_y_series = joblib.load(os.path.join(model_dir, 'series_encoder_y.joblib'))
        
        print("✅ Models loaded successfully!")
        
        # Test with a VIN from the top WMI pattern (9FB - Renault)
        test_vin = "9FB45RC94HM274167"  # This should be in the training data
        
        print(f"\n🎯 Testing VIN: {test_vin}")
        
        # Extract features
        features = extract_vin_features_production(test_vin)
        if not features:
            print("❌ Feature extraction failed")
            return False
        
        print(f"✅ Features extracted: {features}")
        
        try:
            # 1. MAKER PREDICTION - [wmi] only
            print("\n1️⃣ MAKER PREDICTION:")
            X_maker = pd.DataFrame([[features['wmi']]], columns=['wmi'])
            print(f"   Input: {X_maker.values}")
            X_maker_encoded = encoder_x_maker.transform(X_maker)
            print(f"   Encoded shape: {X_maker_encoded.shape}")
            maker_pred_encoded = model_maker.predict(X_maker_encoded)
            predicted_maker = encoder_y_maker.inverse_transform(maker_pred_encoded.reshape(-1, 1))[0][0]
            print(f"   ✅ Predicted Maker: {predicted_maker}")
            
            # 2. MODEL (YEAR) PREDICTION - [year_code] CORRECTED!
            print("\n2️⃣ MODEL (YEAR) PREDICTION:")
            X_model = pd.DataFrame([[features['year_code']]], columns=['year_code'])
            print(f"   Input: {X_model.values}")
            X_model_encoded = encoder_x_model.transform(X_model)
            print(f"   Encoded shape: {X_model_encoded.shape}")
            model_pred_encoded = model_model.predict(X_model_encoded)
            predicted_model = encoder_y_model.inverse_transform(model_pred_encoded.reshape(-1, 1))[0][0]
            print(f"   ✅ Predicted Model (Year): {predicted_model}")
            
            # 3. SERIES PREDICTION - [wmi, vds_full]
            print("\n3️⃣ SERIES PREDICTION:")
            X_series = pd.DataFrame([[features['wmi'], features['vds_full']]], columns=['wmi', 'vds_full'])
            print(f"   Input: {X_series.values}")
            X_series_encoded = encoder_x_series.transform(X_series)
            print(f"   Encoded shape: {X_series_encoded.shape}")
            series_pred_encoded = model_series.predict(X_series_encoded)
            predicted_series = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0][0]
            print(f"   ✅ Predicted Series: {predicted_series}")
            
            print(f"\n🎯 FINAL PREDICTION:")
            print(f"   VIN: {test_vin}")
            print(f"   Maker: {predicted_maker}")
            print(f"   Year: {predicted_model}")
            print(f"   Series: {predicted_series}")
            
            # Test with your original problematic VIN
            print(f"\n" + "="*60)
            print("🔍 TESTING YOUR ORIGINAL VIN (should return Unknown)")
            
            problem_vin = "KMHSH8HX8CU889564"
            print(f"Testing VIN: {problem_vin}")
            
            features2 = extract_vin_features_production(problem_vin)
            if features2:
                print(f"Features: {features2}")
                
                # Try maker prediction
                try:
                    X_maker2 = pd.DataFrame([[features2['wmi']]], columns=['wmi'])
                    X_maker2_encoded = encoder_x_maker.transform(X_maker2)
                    maker_pred2_encoded = model_maker.predict(X_maker2_encoded)
                    predicted_maker2 = encoder_y_maker.inverse_transform(maker_pred2_encoded.reshape(-1, 1))[0][0]
                    print(f"✅ Predicted Maker: {predicted_maker2}")
                except Exception as e:
                    print(f"❌ Maker prediction failed: {e}")
                
                # Try year prediction
                try:
                    X_model2 = pd.DataFrame([[features2['year_code']]], columns=['year_code'])
                    X_model2_encoded = encoder_x_model.transform(X_model2)
                    model_pred2_encoded = model_model.predict(X_model2_encoded)
                    predicted_model2 = encoder_y_model.inverse_transform(model_pred2_encoded.reshape(-1, 1))[0][0]
                    print(f"✅ Predicted Year: {predicted_model2}")
                except Exception as e:
                    print(f"❌ Year prediction failed: {e}")
                    
                # Try series prediction
                try:
                    X_series2 = pd.DataFrame([[features2['wmi'], features2['vds_full']]], columns=['wmi', 'vds_full'])
                    X_series2_encoded = encoder_x_series.transform(X_series2)
                    series_pred2_encoded = model_series.predict(X_series2_encoded)
                    predicted_series2 = encoder_y_series.inverse_transform(series_pred2_encoded.reshape(-1, 1))[0][0]
                    print(f"✅ Predicted Series: {predicted_series2}")
                except Exception as e:
                    print(f"❌ Series prediction failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_specific_vin()
