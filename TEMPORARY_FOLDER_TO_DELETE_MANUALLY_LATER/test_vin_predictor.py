#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VIN Predictor Testing Tool

This tool tests VIN predictions against known database records to identify
accuracy issues and training data problems.
"""

import os
import sys
import sqlite3
import pandas as pd
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_vin_predictions():
    """Test VIN predictions against database records."""
    
    try:
        from unified_consolidado_processor import get_base_path
        from train_vin_predictor import extract_vin_features_production, decode_year
        import joblib
        
        print("🧪 VIN Predictor Testing Tool")
        print("=" * 60)
        
        # Get database and model paths
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        model_dir = os.path.join(base_path, "models")
        
        print(f"📁 Database: {db_path}")
        print(f"📁 Models: {model_dir}")
        print(f"📊 DB exists: {os.path.exists(db_path)}")
        
        # Load models
        print("\n🔄 Loading VIN prediction models...")
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
            
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
        
        # Connect to database
        print("\n🔄 Connecting to database...")
        conn = sqlite3.connect(db_path)
        
        # Test specific VINs mentioned by user
        test_vins = [
            "MALAT41CAJM280395",  # Hyundai VIN from user
            "MM7UR4DF7GW498254",  # Mazda VIN from user
            "TMAJUB1E5C7T35032",  # From database screenshot
            "TMAJUB1E5C7T35539",  # From database screenshot
            "TMAJUB1E5C7T6290",   # From database screenshot
        ]
        
        print(f"\n🎯 Testing {len(test_vins)} specific VINs...")
        
        for vin in test_vins:
            print(f"\n--- Testing VIN: {vin} ---")
            
            # Get actual data from database
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT maker, model, series, COUNT(*) as count
                FROM processed_consolidado 
                WHERE vin_number = ? 
                GROUP BY maker, model, series
                ORDER BY count DESC
            """, (vin,))
            
            actual_records = cursor.fetchall()
            
            if actual_records:
                print(f"📊 Database records for {vin}:")
                for record in actual_records:
                    print(f"  - Maker: {record[0]}, Model: {record[1]}, Series: {record[2]} ({record[3]} records)")
                
                # Use the most common record as ground truth
                actual_maker, actual_model, actual_series = actual_records[0][:3]
            else:
                print(f"❌ VIN {vin} not found in database")
                continue
            
            # Test prediction
            try:
                features = extract_vin_features_production(vin)
                print(f"🔍 VIN Features: WMI={features.get('wmi')}, VDS={features.get('vds')}, Year={features.get('year')}")
                
                # Predict maker
                X_maker = [[features['wmi']]]
                X_maker_encoded = encoder_x_maker.transform(X_maker)
                maker_pred_encoded = model_maker.predict(X_maker_encoded)
                predicted_maker = encoder_y_maker.inverse_transform(maker_pred_encoded.reshape(-1, 1))[0][0]
                
                # Predict model (year)
                X_model = [[features['wmi'], features['vds']]]
                X_model_encoded = encoder_x_model.transform(X_model)
                model_pred_encoded = model_model.predict(X_model_encoded)
                predicted_model = encoder_y_model.inverse_transform(model_pred_encoded.reshape(-1, 1))[0][0]
                
                # Predict series
                X_series = [[features['wmi'], features['vds'], features['year']]]
                X_series_encoded = encoder_x_series.transform(X_series)
                series_pred_encoded = model_series.predict(X_series_encoded)
                predicted_series = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0][0]
                
                print(f"🤖 Predictions:")
                print(f"  - Maker: {predicted_maker} {'✅' if predicted_maker == actual_maker else '❌'}")
                print(f"  - Model: {predicted_model} {'✅' if str(predicted_model) == str(actual_model) else '❌'}")
                print(f"  - Series: {predicted_series} {'✅' if predicted_series == actual_series else '❌'}")
                
                print(f"📋 Actual:")
                print(f"  - Maker: {actual_maker}")
                print(f"  - Model: {actual_model}")
                print(f"  - Series: {actual_series}")
                
            except Exception as e:
                print(f"❌ Prediction error: {e}")
        
        # Analyze WMI coverage
        print(f"\n📊 Analyzing WMI Coverage...")
        cursor.execute("""
            SELECT SUBSTR(vin_number, 1, 3) as wmi, maker, COUNT(*) as count
            FROM processed_consolidado 
            GROUP BY wmi, maker
            ORDER BY count DESC
            LIMIT 20
        """)
        
        wmi_data = cursor.fetchall()
        print("Top 20 WMI patterns in database:")
        for wmi, maker, count in wmi_data:
            print(f"  {wmi}: {maker} ({count:,} records)")
        
        # Check specific problematic WMIs
        problematic_wmis = ["MAL", "MM7"]  # From user's examples
        print(f"\n🔍 Checking problematic WMIs...")
        for wmi in problematic_wmis:
            cursor.execute("""
                SELECT maker, COUNT(*) as count
                FROM processed_consolidado 
                WHERE SUBSTR(vin_number, 1, 3) = ?
                GROUP BY maker
                ORDER BY count DESC
            """, (wmi,))
            
            results = cursor.fetchall()
            if results:
                print(f"WMI '{wmi}' in database:")
                for maker, count in results:
                    print(f"  - {maker}: {count:,} records")
            else:
                print(f"WMI '{wmi}' NOT found in database")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during VIN testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vin_predictions()
    
    if success:
        print("\n🎉 VIN testing completed!")
    else:
        print("\n💥 VIN testing failed!")
