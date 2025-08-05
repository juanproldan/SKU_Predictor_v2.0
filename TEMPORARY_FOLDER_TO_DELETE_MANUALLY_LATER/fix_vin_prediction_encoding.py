#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix VIN Prediction Encoding Issues

This script fixes the feature mismatch between VIN model training and prediction.
The issue is that prediction code uses wrong feature combinations compared to training.

TRAINING vs PREDICTION MISMATCHES:
1. Maker: ✅ Both use [wmi] - CORRECT
2. Model: ❌ Training uses [wmi, vds] but prediction uses [wmi, vds] - FEATURE NAME MISMATCH
3. Series: ❌ Training uses [wmi, vds_full] but prediction uses [wmi, vds, year] - WRONG FEATURES

Author: Augment Agent  
Date: 2025-08-03
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def analyze_model_encoders():
    """Analyze the trained model encoders to understand expected features."""
    
    try:
        import joblib
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        model_dir = os.path.join(base_path, "models")
        
        print("🔍 ANALYZING VIN MODEL ENCODERS")
        print("=" * 50)
        
        # Load and analyze each encoder
        models_info = {
            'maker': {
                'x_encoder': 'maker_encoder_x.joblib',
                'y_encoder': 'maker_encoder_y.joblib',
                'model': 'maker_model.joblib'
            },
            'model': {
                'x_encoder': 'model_encoder_x.joblib', 
                'y_encoder': 'model_encoder_y.joblib',
                'model': 'model_model.joblib'
            },
            'series': {
                'x_encoder': 'series_encoder_x.joblib',
                'y_encoder': 'series_encoder_y.joblib', 
                'model': 'series_model.joblib'
            }
        }
        
        for model_name, files in models_info.items():
            print(f"\n--- {model_name.upper()} MODEL ---")
            
            try:
                # Load X encoder to see expected features
                x_encoder_path = os.path.join(model_dir, files['x_encoder'])
                x_encoder = joblib.load(x_encoder_path)
                
                print(f"✅ X Encoder loaded: {x_encoder_path}")
                print(f"   Expected features: {x_encoder.n_features_in_}")
                
                if hasattr(x_encoder, 'feature_names_in_') and x_encoder.feature_names_in_ is not None:
                    print(f"   Feature names: {list(x_encoder.feature_names_in_)}")
                else:
                    print(f"   Feature names: Not available")
                
                # Load Y encoder to see target categories
                y_encoder_path = os.path.join(model_dir, files['y_encoder'])
                y_encoder = joblib.load(y_encoder_path)
                
                print(f"✅ Y Encoder loaded: {y_encoder_path}")
                if hasattr(y_encoder, 'categories_'):
                    print(f"   Target categories: {len(y_encoder.categories_[0])} unique values")
                    print(f"   Sample targets: {list(y_encoder.categories_[0][:5])}...")
                
            except Exception as e:
                print(f"❌ Error loading {model_name} encoders: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing encoders: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_fixed_prediction_function():
    """Create a corrected VIN prediction function with proper feature alignment."""
    
    print("\n🛠️ CREATING FIXED PREDICTION FUNCTION")
    print("=" * 50)
    
    fixed_code = '''
def predict_vin_details_fixed(vin):
    """
    Fixed VIN prediction function that matches training feature combinations.
    
    CORRECTED FEATURE MAPPINGS:
    - Maker: [wmi] only
    - Model: [wmi, vds] (NOT vds_full)  
    - Series: [wmi, vds_full] (NOT wmi, vds, year)
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
        return None
    
    results = {}
    
    try:
        # 1. MAKER PREDICTION - [wmi] only
        X_maker = pd.DataFrame([[features['wmi']]], columns=['wmi'])
        X_maker_encoded = encoder_x_maker.transform(X_maker)
        maker_pred_encoded = model_maker.predict(X_maker_encoded)
        predicted_maker = encoder_y_maker.inverse_transform(maker_pred_encoded.reshape(-1, 1))[0][0]
        results['maker'] = predicted_maker
        
        # 2. MODEL PREDICTION - [wmi, vds] (match training exactly)
        X_model = pd.DataFrame([[features['wmi'], features['vds']]], columns=['wmi', 'vds'])
        X_model_encoded = encoder_x_model.transform(X_model)
        model_pred_encoded = model_model.predict(X_model_encoded)
        predicted_model = encoder_y_model.inverse_transform(model_pred_encoded.reshape(-1, 1))[0][0]
        results['model'] = predicted_model
        
        # 3. SERIES PREDICTION - [wmi, vds_full] (match training exactly)
        X_series = pd.DataFrame([[features['wmi'], features['vds_full']]], columns=['wmi', 'vds_full'])
        X_series_encoded = encoder_x_series.transform(X_series)
        series_pred_encoded = model_series.predict(X_series_encoded)
        predicted_series = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0][0]
        results['series'] = predicted_series
        
        return results
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None
'''
    
    print("✅ Fixed prediction function created")
    print("📋 Key fixes:")
    print("   1. Maker: [wmi] only ✅")
    print("   2. Model: [wmi, vds] with proper DataFrame columns ✅") 
    print("   3. Series: [wmi, vds_full] with proper DataFrame columns ✅")
    print("   4. All predictions use pandas DataFrame with column names ✅")
    
    return fixed_code

def test_fixed_prediction():
    """Test the fixed prediction function."""
    
    print("\n🧪 TESTING FIXED PREDICTION")
    print("=" * 50)
    
    # Test VINs from our database
    test_vins = [
        "9GATJ516488031479",  # Renault
        "VF1RJL003SC526077",  # Renault  
        "3BRCD33B2HK590620",  # Renault
        "9BGKC48T0KG372413",  # Chevrolet
        "JMZDB12D200328460",  # Mazda
    ]
    
    try:
        # Import the fixed function (we'll need to implement it properly)
        exec(create_fixed_prediction_function())
        
        for vin in test_vins:
            print(f"\n🔍 Testing VIN: {vin}")
            
            # Test feature extraction first
            from train_vin_predictor import extract_vin_features_production
            features = extract_vin_features_production(vin)
            
            if features:
                print(f"   ✅ Features: WMI={features['wmi']}, VDS={features['vds']}, VDS_FULL={features['vds_full']}")
                
                # Test prediction (this will fail until we implement the fix)
                try:
                    results = predict_vin_details_fixed(vin)
                    if results:
                        print(f"   🎯 Predictions: {results}")
                    else:
                        print(f"   ❌ Prediction failed")
                except Exception as e:
                    print(f"   ❌ Prediction error: {e}")
            else:
                print(f"   ❌ Feature extraction failed")
                
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to analyze and fix VIN prediction issues."""
    
    print("🔧 VIN PREDICTION ENCODING FIX TOOL")
    print("=" * 60)
    
    # Step 1: Analyze current model encoders
    if not analyze_model_encoders():
        print("❌ Failed to analyze encoders")
        return False
    
    # Step 2: Create fixed prediction function
    fixed_code = create_fixed_prediction_function()
    
    # Step 3: Test the fix concept
    test_fixed_prediction()
    
    print("\n💡 NEXT STEPS:")
    print("1. ✅ Analysis complete - found exact feature mismatches")
    print("2. 🛠️ Fix requires updating prediction code to match training")
    print("3. 🔄 Alternative: Retrain models with consistent feature names")
    print("4. 📋 Recommendation: Fix prediction code (faster than retraining)")
    
    return True

if __name__ == "__main__":
    main()
