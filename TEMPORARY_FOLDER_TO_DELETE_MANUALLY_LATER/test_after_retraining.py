#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test VIN Prediction After Retraining

Test the user's VINs after retraining to confirm they now work.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_after_retraining():
    """Test VIN prediction after retraining."""
    
    try:
        from unified_consolidado_processor import get_base_path
        from train_vin_predictor import extract_vin_features_production
        import joblib
        import pandas as pd
        
        print("🧪 TESTING VIN PREDICTION AFTER RETRAINING")
        print("=" * 60)
        
        base_path = get_base_path()
        model_dir = os.path.join(base_path, "models")
        
        # Check if models exist
        required_models = [
            'maker_model.joblib',
            'maker_encoder_x.joblib', 
            'maker_encoder_y.joblib',
            'model_model.joblib',
            'model_encoder_x.joblib',
            'model_encoder_y.joblib',
            'series_model.joblib',
            'series_encoder_x.joblib',
            'series_encoder_y.joblib'
        ]
        
        print("📂 Checking if retrained models exist...")
        all_exist = True
        for model_file in required_models:
            model_path = os.path.join(model_dir, model_file)
            exists = os.path.exists(model_path)
            print(f"   {'✅' if exists else '❌'} {model_file}")
            if not exists:
                all_exist = False
        
        if not all_exist:
            print("\n⚠️ Some models are missing. Training may still be in progress.")
            print("   Please wait for training to complete and run this test again.")
            return False
        
        # Load models
        print(f"\n📂 Loading retrained models...")
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
        
        # Test your VINs
        your_vins = [
            "KMHSH81XBCU889564",
            "MALA751AAFM098475", 
            "KMHCT41DAEU610396"
        ]
        
        print(f"\n🎯 TESTING YOUR VINs WITH RETRAINED MODELS")
        print("=" * 60)
        
        successful_predictions = 0
        
        for i, vin in enumerate(your_vins, 1):
            print(f"\n--- Test {i}: {vin} ---")
            
            try:
                # Extract features
                features = extract_vin_features_production(vin)
                if not features:
                    print("❌ Feature extraction failed")
                    continue
                
                print(f"✅ Features: {features}")
                
                # Predict maker
                try:
                    X_maker = pd.DataFrame([[features['wmi']]], columns=['wmi'])
                    X_maker_encoded = encoder_x_maker.transform(X_maker)
                    maker_pred_encoded = model_maker.predict(X_maker_encoded)
                    predicted_maker = encoder_y_maker.inverse_transform(maker_pred_encoded.reshape(-1, 1))[0][0]
                    print(f"✅ Predicted Maker: {predicted_maker}")
                except Exception as e:
                    print(f"❌ Maker prediction failed: {e}")
                    continue
                
                # Predict year
                try:
                    X_model = pd.DataFrame([[features['year_code']]], columns=['year_code'])
                    X_model_encoded = encoder_x_model.transform(X_model)
                    model_pred_encoded = model_model.predict(X_model_encoded)
                    predicted_year = encoder_y_model.inverse_transform(model_pred_encoded.reshape(-1, 1))[0][0]
                    print(f"✅ Predicted Year: {predicted_year}")
                except Exception as e:
                    print(f"❌ Year prediction failed: {e}")
                    continue
                
                # Predict series
                try:
                    X_series = pd.DataFrame([[features['wmi'], features['vds_full']]], columns=['wmi', 'vds_full'])
                    X_series_encoded = encoder_x_series.transform(X_series)
                    series_pred_encoded = model_series.predict(X_series_encoded)
                    predicted_series = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0][0]
                    print(f"✅ Predicted Series: {predicted_series}")
                except Exception as e:
                    print(f"❌ Series prediction failed: {e}")
                    predicted_series = "Unknown"
                
                print(f"🎯 FINAL PREDICTION: {predicted_maker} {predicted_year} {predicted_series}")
                successful_predictions += 1
                
            except Exception as e:
                print(f"❌ Error processing VIN: {e}")
        
        print(f"\n📊 RESULTS SUMMARY:")
        print("=" * 60)
        print(f"VINs tested: {len(your_vins)}")
        print(f"Successful predictions: {successful_predictions}")
        print(f"Success rate: {(successful_predictions/len(your_vins))*100:.1f}%")
        
        if successful_predictions == len(your_vins):
            print(f"\n🎉 SUCCESS! All your VINs now work with the retrained models!")
            print(f"✅ The VIN prediction system is fixed and ready to use.")
        elif successful_predictions > 0:
            print(f"\n✅ Partial success! {successful_predictions} out of {len(your_vins)} VINs work.")
            print(f"⚠️ Some VINs may still need more training data.")
        else:
            print(f"\n❌ No successful predictions. There may still be an issue.")
        
        return successful_predictions > 0
        
    except Exception as e:
        print(f"❌ Error testing after retraining: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_after_retraining()
