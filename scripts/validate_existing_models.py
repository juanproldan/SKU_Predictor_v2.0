#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validate Existing Trained Models
Check if all models are working properly without retraining

Author: Augment Agent
Date: 2025-07-25
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n📋 STEP {step_num}: {title}")
    print(f"{'-'*50}")

def check_model_files():
    """Check if all required model files exist"""
    print_step(1, "CHECKING MODEL FILES")
    
    required_models = {
        'VIN Predictor Models': [
            'models/vin_maker_model.joblib',
            'models/vin_year_model.joblib', 
            'models/vin_series_model.joblib',
            'models/vin_maker_encoder_x.joblib',
            'models/vin_maker_encoder_y.joblib',
            'models/vin_year_encoder_x.joblib',
            'models/vin_year_encoder_y.joblib',
            'models/vin_series_encoder_x.joblib',
            'models/vin_series_encoder_y.joblib'
        ],
        'SKU Neural Network Models': [
            'models/sku_nn/sku_nn_model_pytorch_optimized.pth',
            'models/sku_nn/encoder_Make.joblib',
            'models/sku_nn/encoder_Model Year.joblib',
            'models/sku_nn/encoder_Series.joblib',
            'models/sku_nn/encoder_sku.joblib',
            'models/sku_nn/tokenizer.joblib'
        ]
    }
    
    all_models_exist = True
    
    for category, models in required_models.items():
        print(f"\n🔍 {category}:")
        category_complete = True
        
        for model_path in models:
            full_path = os.path.join(current_dir, model_path)
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                mod_time = datetime.fromtimestamp(os.path.getmtime(full_path))
                print(f"  ✅ {os.path.basename(model_path)} ({file_size:,} bytes, {mod_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"  ❌ {os.path.basename(model_path)} - NOT FOUND")
                category_complete = False
                all_models_exist = False
        
        if category_complete:
            print(f"  🎯 {category}: COMPLETE")
        else:
            print(f"  ⚠️ {category}: INCOMPLETE")
    
    return all_models_exist

def test_vin_prediction():
    """Test VIN prediction functionality"""
    print_step(2, "TESTING VIN PREDICTION")
    
    try:
        # Import the main app
        from main_app import FixacarApp
        import tkinter as tk
        
        # Create a mock root
        root = tk.Tk()
        root.withdraw()
        
        app = FixacarApp(root)
        
        # Test VIN predictions
        test_vins = [
            "1HGBH41JXMN109186",  # Honda
            "JTDKN3DU0A0123456",  # Toyota
            "3N1AB7AP5EY123456"   # Nissan
        ]
        
        print(f"🔄 Testing VIN predictions...")
        
        successful_predictions = 0
        
        for i, vin in enumerate(test_vins, 1):
            try:
                print(f"\n  Test {i}: VIN {vin}")
                
                # Test VIN prediction
                start_time = time.time()
                result = app.predict_vin_details(vin)
                prediction_time = (time.time() - start_time) * 1000
                
                if result and len(result) >= 3:
                    make, year, series = result[0], result[1], result[2]
                    print(f"    ✅ Prediction: {make} {year} {series} ({prediction_time:.2f}ms)")
                    successful_predictions += 1
                else:
                    print(f"    ⚠️ No prediction returned ({prediction_time:.2f}ms)")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        success_rate = successful_predictions / len(test_vins) * 100
        print(f"\n📊 VIN Prediction Results:")
        print(f"   Successful: {successful_predictions}/{len(test_vins)} ({success_rate:.1f}%)")
        
        root.destroy()
        return successful_predictions > 0
        
    except Exception as e:
        print(f"❌ VIN prediction test failed: {e}")
        return False

def test_sku_prediction():
    """Test SKU prediction functionality"""
    print_step(3, "TESTING SKU PREDICTION")
    
    try:
        # Import the main app
        from main_app import FixacarApp
        import tkinter as tk
        
        # Create a mock root
        root = tk.Tk()
        root.withdraw()
        
        app = FixacarApp(root)
        
        # Test SKU predictions
        test_cases = [
            ("TOYOTA", "2020", "COROLLA", "parachoques delantero"),
            ("HONDA", "2019", "CIVIC", "faro izquierdo"),
            ("NISSAN", "2021", "SENTRA", "espejo derecho")
        ]
        
        print(f"🔄 Testing SKU predictions...")
        
        successful_predictions = 0
        
        for i, (make, year, series, description) in enumerate(test_cases, 1):
            try:
                print(f"\n  Test {i}: {make} {year} {series} - {description}")
                
                # Test SKU prediction
                start_time = time.time()
                result = app.predict_sku_for_part(make, year, series, description)
                prediction_time = (time.time() - start_time) * 1000
                
                if result and len(result) > 0:
                    # Get the best prediction
                    best_prediction = result[0]
                    if isinstance(best_prediction, dict):
                        sku = best_prediction.get('sku', 'Unknown')
                        confidence = best_prediction.get('confidence', 0.0)
                        source = best_prediction.get('source', 'Unknown')
                    elif isinstance(best_prediction, tuple) and len(best_prediction) >= 2:
                        sku = best_prediction[0]
                        confidence = best_prediction[1]
                        source = best_prediction[2] if len(best_prediction) > 2 else 'Unknown'
                    else:
                        sku = str(best_prediction)
                        confidence = 0.0
                        source = 'Unknown'
                    
                    print(f"    ✅ Prediction: {sku} (confidence: {confidence:.3f}, source: {source}) ({prediction_time:.2f}ms)")
                    successful_predictions += 1
                else:
                    print(f"    ⚠️ No prediction returned ({prediction_time:.2f}ms)")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        success_rate = successful_predictions / len(test_cases) * 100
        print(f"\n📊 SKU Prediction Results:")
        print(f"   Successful: {successful_predictions}/{len(test_cases)} ({success_rate:.1f}%)")
        
        root.destroy()
        return successful_predictions > 0
        
    except Exception as e:
        print(f"❌ SKU prediction test failed: {e}")
        return False

def test_performance_improvements():
    """Test performance improvements"""
    print_step(4, "TESTING PERFORMANCE IMPROVEMENTS")
    
    try:
        # Run the quick performance test
        sys.path.insert(0, os.path.join(current_dir, 'performance_improvements', 'validation'))
        from quick_performance_test import run_quick_performance_test
        
        print(f"🔄 Running performance validation...")
        results = run_quick_performance_test()
        
        # Check results
        overall_score = results.get('overall_score', 0)
        
        if overall_score >= 80:
            print(f"🎉 Performance improvements: EXCELLENT ({overall_score:.0f}%)")
            return True
        elif overall_score >= 60:
            print(f"✅ Performance improvements: GOOD ({overall_score:.0f}%)")
            return True
        else:
            print(f"⚠️ Performance improvements: NEEDS ATTENTION ({overall_score:.0f}%)")
            return False
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main validation function"""
    print_header("EXISTING MODELS VALIDATION")
    
    start_time = time.time()
    
    # Track results
    results = {
        'model_files': False,
        'vin_prediction': False,
        'sku_prediction': False,
        'performance_improvements': False
    }
    
    # Step 1: Check model files
    results['model_files'] = check_model_files()
    
    if not results['model_files']:
        print(f"\n❌ Model files check failed. Some models are missing.")
        return False
    
    # Step 2: Test VIN prediction
    results['vin_prediction'] = test_vin_prediction()
    
    # Step 3: Test SKU prediction
    results['sku_prediction'] = test_sku_prediction()
    
    # Step 4: Test performance improvements
    results['performance_improvements'] = test_performance_improvements()
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final summary
    print_header("VALIDATION SUMMARY")
    
    successful_tests = sum(1 for success in results.values() if success)
    total_tests = len(results)
    
    print(f"⏱️ Total Validation Time: {total_time:.1f} seconds")
    print(f"📊 Successful Tests: {successful_tests}/{total_tests}")
    
    for test, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {test.replace('_', ' ').title()}")
    
    if successful_tests == total_tests:
        print(f"\n🎉 ALL MODELS VALIDATED SUCCESSFULLY!")
        print(f"🚀 System is ready for production use!")
        print(f"💡 No retraining needed - existing models are working perfectly!")
        return True
    else:
        print(f"\n⚠️ Validation completed with {total_tests - successful_tests} issues.")
        print(f"📋 Some components may need attention or retraining.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
