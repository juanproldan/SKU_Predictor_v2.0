#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive VIN Prediction Testing

This script tests VIN prediction with 5 VINs from each major maker,
covering different series to evaluate prediction accuracy and coverage.
"""

import os
import sys
import sqlite3
import random
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def get_test_vins_by_maker():
    """Get 5 VINs from each major maker, preferably different series."""
    
    try:
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔍 Finding VINs by maker and series...")
        
        # Get top makers by VIN count (using vin_number column)
        cursor.execute("""
            SELECT maker, COUNT(DISTINCT vin_number) as vin_count
            FROM processed_consolidado
            WHERE vin_number IS NOT NULL
            AND vin_number != ''
            AND LENGTH(vin_number) = 17
            AND vin_number != '00000000000000000'
            GROUP BY maker
            ORDER BY vin_count DESC
            LIMIT 10
        """)
        
        top_makers = cursor.fetchall()
        print(f"📊 Top makers by VIN count:")
        for maker, count in top_makers:
            print(f"   {maker}: {count:,} VINs")
        
        test_vins = {}
        
        for maker, _ in top_makers:
            print(f"\n🔍 Getting VINs for {maker}...")
            
            # Get VINs grouped by series for this maker (using vin_number column)
            cursor.execute("""
                SELECT vin_number, series, model, COUNT(*) as frequency
                FROM processed_consolidado
                WHERE maker = ?
                AND vin_number IS NOT NULL
                AND vin_number != ''
                AND LENGTH(vin_number) = 17
                AND vin_number != '00000000000000000'
                AND series IS NOT NULL
                AND series != ''
                GROUP BY vin_number, series, model
                ORDER BY series, frequency DESC
            """, (maker,))
            
            maker_vins = cursor.fetchall()
            
            if not maker_vins:
                print(f"   ❌ No VINs found for {maker}")
                continue
            
            # Group by series
            series_vins = defaultdict(list)
            for vin, series, model, freq in maker_vins:
                series_vins[series].append((vin, model, freq))
            
            print(f"   📋 Found {len(series_vins)} series with VINs")
            
            # Select up to 5 VINs, preferring different series
            selected_vins = []
            series_list = list(series_vins.keys())
            
            # Try to get one VIN from each series (up to 5)
            for i, series in enumerate(series_list[:5]):
                if len(selected_vins) >= 5:
                    break
                    
                # Get the most frequent VIN from this series
                series_vin_data = series_vins[series]
                vin, model, freq = series_vin_data[0]  # Most frequent
                
                selected_vins.append({
                    'vin': vin,
                    'maker': maker,
                    'series': series,
                    'model': model,
                    'frequency': freq
                })
                
                print(f"   ✅ {series[:30]}... | {vin} | {model} | freq: {freq}")
            
            # If we need more VINs and have fewer series, get more from existing series
            if len(selected_vins) < 5:
                for series in series_list:
                    if len(selected_vins) >= 5:
                        break
                    
                    # Skip if we already have a VIN from this series
                    if any(v['series'] == series for v in selected_vins):
                        continue
                    
                    # Get additional VINs from this series
                    for vin, model, freq in series_vins[series][1:]:  # Skip first (already used)
                        if len(selected_vins) >= 5:
                            break
                        
                        selected_vins.append({
                            'vin': vin,
                            'maker': maker,
                            'series': series,
                            'model': model,
                            'frequency': freq
                        })
                        
                        print(f"   ✅ {series[:30]}... | {vin} | {model} | freq: {freq}")
            
            test_vins[maker] = selected_vins
            print(f"   🎯 Selected {len(selected_vins)} VINs for {maker}")
        
        conn.close()
        return test_vins
        
    except Exception as e:
        print(f"❌ Error getting test VINs: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_vin_predictions(test_vins):
    """Test VIN predictions for all collected VINs."""
    
    try:
        from train_vin_predictor import extract_vin_features_production
        import joblib
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        model_dir = os.path.join(base_path, "models")
        
        print("\n🧪 Testing VIN Predictions")
        print("=" * 60)
        
        # Try to load VIN prediction models
        models_available = False
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
            
            models_available = True
            print("✅ VIN prediction models loaded successfully")
        except Exception as e:
            print(f"❌ VIN models not available: {e}")
            return False
        
        # Test statistics
        total_tests = 0
        successful_extractions = 0
        successful_predictions = 0
        maker_matches = 0
        year_matches = 0
        series_matches = 0
        
        results_by_maker = {}
        
        for maker, vins in test_vins.items():
            print(f"\n--- Testing {maker} ({len(vins)} VINs) ---")
            
            maker_results = {
                'total': len(vins),
                'extraction_success': 0,
                'prediction_success': 0,
                'maker_match': 0,
                'year_match': 0,
                'series_match': 0,
                'details': []
            }
            
            for i, vin_data in enumerate(vins, 1):
                vin = vin_data['vin']
                expected_maker = vin_data['maker']
                expected_series = vin_data['series']
                expected_year = str(vin_data['model'])
                
                print(f"\n{i}. VIN: {vin}")
                print(f"   Expected: {expected_maker} {expected_year} {expected_series[:40]}...")
                
                total_tests += 1
                
                # Test feature extraction
                try:
                    features = extract_vin_features_production(vin)
                    
                    if features:
                        successful_extractions += 1
                        maker_results['extraction_success'] += 1
                        
                        print(f"   ✅ Features: WMI={features['wmi']}, Year={features['year']}")
                        
                        # Test predictions
                        try:
                            # Predict maker - [wmi] only ✅
                            import pandas as pd
                            X_maker = pd.DataFrame([[features['wmi']]], columns=['wmi'])
                            X_maker_encoded = encoder_x_maker.transform(X_maker)
                            maker_pred_encoded = model_maker.predict(X_maker_encoded)
                            predicted_maker = encoder_y_maker.inverse_transform(maker_pred_encoded.reshape(-1, 1))[0][0]

                            # Predict year - [year_code] only ✅ (CORRECTED!)
                            X_model = pd.DataFrame([[features['year_code']]], columns=['year_code'])
                            X_model_encoded = encoder_x_model.transform(X_model)
                            model_pred_encoded = model_model.predict(X_model_encoded)
                            predicted_year = encoder_y_model.inverse_transform(model_pred_encoded.reshape(-1, 1))[0][0]

                            # Predict series - [wmi, vds_full] only ✅ (CORRECTED!)
                            X_series = pd.DataFrame([[features['wmi'], features['vds_full']]], columns=['wmi', 'vds_full'])
                            X_series_encoded = encoder_x_series.transform(X_series)
                            series_pred_encoded = model_series.predict(X_series_encoded)
                            predicted_series = encoder_y_series.inverse_transform(series_pred_encoded.reshape(-1, 1))[0][0]
                            
                            successful_predictions += 1
                            maker_results['prediction_success'] += 1
                            
                            print(f"   🎯 Predicted: {predicted_maker} {predicted_year} {str(predicted_series)[:40]}...")
                            
                            # Check matches
                            maker_match = str(expected_maker).upper() in str(predicted_maker).upper() or str(predicted_maker).upper() in str(expected_maker).upper()
                            year_match = str(predicted_year) == expected_year
                            series_match = str(predicted_series) == expected_series
                            
                            if maker_match:
                                maker_matches += 1
                                maker_results['maker_match'] += 1
                            if year_match:
                                year_matches += 1
                                maker_results['year_match'] += 1
                            if series_match:
                                series_matches += 1
                                maker_results['series_match'] += 1
                            
                            print(f"   📊 Matches: Maker={'✅' if maker_match else '❌'} Year={'✅' if year_match else '❌'} Series={'✅' if series_match else '❌'}")
                            
                            maker_results['details'].append({
                                'vin': vin,
                                'extraction': True,
                                'prediction': True,
                                'maker_match': maker_match,
                                'year_match': year_match,
                                'series_match': series_match,
                                'predicted_maker': predicted_maker,
                                'predicted_year': predicted_year,
                                'predicted_series': str(predicted_series)[:50]
                            })
                            
                        except Exception as e:
                            print(f"   ❌ Prediction error: {e}")
                            maker_results['details'].append({
                                'vin': vin,
                                'extraction': True,
                                'prediction': False,
                                'error': str(e)
                            })
                    else:
                        print(f"   ❌ Feature extraction failed")
                        maker_results['details'].append({
                            'vin': vin,
                            'extraction': False,
                            'prediction': False
                        })
                        
                except Exception as e:
                    print(f"   ❌ Extraction error: {e}")
                    maker_results['details'].append({
                        'vin': vin,
                        'extraction': False,
                        'prediction': False,
                        'error': str(e)
                    })
            
            results_by_maker[maker] = maker_results
            
            # Maker summary
            print(f"\n📊 {maker} Summary:")
            print(f"   Extraction: {maker_results['extraction_success']}/{maker_results['total']} ({maker_results['extraction_success']/maker_results['total']*100:.1f}%)")
            print(f"   Prediction: {maker_results['prediction_success']}/{maker_results['total']} ({maker_results['prediction_success']/maker_results['total']*100:.1f}%)")
            if maker_results['prediction_success'] > 0:
                print(f"   Maker Match: {maker_results['maker_match']}/{maker_results['prediction_success']} ({maker_results['maker_match']/maker_results['prediction_success']*100:.1f}%)")
                print(f"   Year Match: {maker_results['year_match']}/{maker_results['prediction_success']} ({maker_results['year_match']/maker_results['prediction_success']*100:.1f}%)")
                print(f"   Series Match: {maker_results['series_match']}/{maker_results['prediction_success']} ({maker_results['series_match']/maker_results['prediction_success']*100:.1f}%)")
        
        # Overall summary
        print(f"\n🎯 OVERALL RESULTS")
        print("=" * 40)
        print(f"Total VINs tested: {total_tests}")
        print(f"Feature extraction success: {successful_extractions}/{total_tests} ({successful_extractions/total_tests*100:.1f}%)")
        print(f"Prediction success: {successful_predictions}/{total_tests} ({successful_predictions/total_tests*100:.1f}%)")
        
        if successful_predictions > 0:
            print(f"Maker accuracy: {maker_matches}/{successful_predictions} ({maker_matches/successful_predictions*100:.1f}%)")
            print(f"Year accuracy: {year_matches}/{successful_predictions} ({year_matches/successful_predictions*100:.1f}%)")
            print(f"Series accuracy: {series_matches}/{successful_predictions} ({series_matches/successful_predictions*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during VIN testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run comprehensive VIN testing."""
    
    print("🚗 Comprehensive VIN Prediction Testing")
    print("=" * 50)
    
    # Get test VINs
    test_vins = get_test_vins_by_maker()
    
    if not test_vins:
        print("❌ No test VINs found")
        return False
    
    print(f"\n✅ Collected VINs from {len(test_vins)} makers")
    total_vins = sum(len(vins) for vins in test_vins.values())
    print(f"📊 Total VINs to test: {total_vins}")
    
    # Test predictions
    success = test_vin_predictions(test_vins)
    
    if success:
        print("\n🎉 Comprehensive VIN testing completed!")
    else:
        print("\n💥 VIN testing failed!")
    
    return success

if __name__ == "__main__":
    main()
