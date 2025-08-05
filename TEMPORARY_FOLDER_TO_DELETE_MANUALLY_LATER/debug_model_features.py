#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Model Feature Names

This script examines what feature names the trained models expect
vs what the prediction code is providing.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_model_features():
    """Debug the feature names expected by trained models."""
    
    try:
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        model_dir = os.path.join(base_path, "models")
        
        print("🔍 DEBUGGING MODEL FEATURE EXPECTATIONS")
        print("=" * 60)
        
        # Load and examine each encoder
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
                
                print(f"✅ X Encoder loaded: {files['x_encoder']}")
                print(f"   Expected features count: {x_encoder.n_features_in_}")
                
                if hasattr(x_encoder, 'feature_names_in_') and x_encoder.feature_names_in_ is not None:
                    print(f"   Expected feature names: {list(x_encoder.feature_names_in_)}")
                else:
                    print(f"   Feature names: Not stored in encoder")
                
                # Check if it's a OneHotEncoder or similar
                if hasattr(x_encoder, 'categories_'):
                    print(f"   Categories per feature: {[len(cat) for cat in x_encoder.categories_]}")
                    for i, cat in enumerate(x_encoder.categories_):
                        print(f"     Feature {i}: {len(cat)} categories, samples: {list(cat[:5])}")
                
                # Load Y encoder to see target categories
                y_encoder_path = os.path.join(model_dir, files['y_encoder'])
                y_encoder = joblib.load(y_encoder_path)
                
                print(f"✅ Y Encoder loaded: {files['y_encoder']}")
                if hasattr(y_encoder, 'categories_'):
                    print(f"   Target categories: {len(y_encoder.categories_[0])} unique values")
                    print(f"   Sample targets: {list(y_encoder.categories_[0][:10])}")
                
            except Exception as e:
                print(f"❌ Error loading {model_name} encoders: {e}")
        
        # Now check what the prediction code is trying to send
        print(f"\n🔍 CHECKING PREDICTION CODE FEATURE GENERATION:")
        print("=" * 60)
        
        from train_vin_predictor import extract_vin_features_production
        
        # Test with a sample VIN
        test_vin = "9FB45RC94HM274167"
        features = extract_vin_features_production(test_vin)
        
        if features:
            print(f"Sample VIN: {test_vin}")
            print(f"Generated features: {features}")
            print(f"Feature keys: {list(features.keys())}")
        else:
            print("❌ Feature extraction failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error debugging model features: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_model_features()
