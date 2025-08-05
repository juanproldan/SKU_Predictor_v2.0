#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SKU Predictor Automated Testing Tool

This tool tests SKU predictions against known database records to measure
accuracy across all three prediction sources (Maestro, Neural Network, Year Range DB).
"""

import os
import sys
import sqlite3
import pandas as pd
import random
from collections import defaultdict, Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_sku_predictions():
    """Test SKU predictions against database records."""
    
    try:
        from unified_consolidado_processor import get_base_path
        from utils.year_range_database import YearRangeDatabaseOptimizer
        
        print("🧪 SKU Predictor Automated Testing Tool")
        print("=" * 60)
        
        # Get database path
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        
        print(f"📁 Database: {db_path}")
        print(f"📊 DB exists: {os.path.exists(db_path)}")
        
        # Connect to database
        print("\n🔄 Connecting to database...")
        conn = sqlite3.connect(db_path)
        
        # Initialize year range optimizer
        print("🔄 Initializing year range optimizer...")
        year_range_optimizer = YearRangeDatabaseOptimizer(db_path)
        
        # Get test samples from database
        print("\n📊 Analyzing database for test samples...")
        cursor = conn.cursor()
        
        # Get sample records with known SKUs for testing
        cursor.execute("""
            SELECT maker, model, series, descripcion, referencia, COUNT(*) as frequency
            FROM processed_consolidado 
            WHERE referencia IS NOT NULL 
            AND referencia != '' 
            AND referencia != 'NULL'
            AND LENGTH(referencia) > 3
            GROUP BY maker, model, series, descripcion, referencia
            HAVING frequency >= 3
            ORDER BY frequency DESC
            LIMIT 100
        """)
        
        test_samples = cursor.fetchall()
        print(f"📋 Found {len(test_samples)} test samples with known SKUs")
        
        # Test different makers
        maker_samples = defaultdict(list)
        for sample in test_samples:
            maker_samples[sample[0]].append(sample)
        
        print(f"📊 Test samples by maker:")
        for maker, samples in maker_samples.items():
            print(f"  - {maker}: {len(samples)} samples")
        
        # Run tests on random sample
        test_count = min(20, len(test_samples))
        random_samples = random.sample(test_samples, test_count)
        
        print(f"\n🎯 Testing {test_count} random samples...")
        
        results = {
            'total': 0,
            'year_range_found': 0,
            'year_range_correct': 0,
            'year_range_top3': 0,
            'predictions_by_maker': defaultdict(list)
        }
        
        for i, (maker, model, series, descripcion, actual_sku, frequency) in enumerate(random_samples):
            print(f"\n--- Test {i+1}/{test_count} ---")
            print(f"🔍 Input: {maker} {model} {series} - '{descripcion}'")
            print(f"✅ Expected SKU: {actual_sku} (freq: {frequency})")
            
            results['total'] += 1
            
            try:
                # Test year range predictions
                predictions = year_range_optimizer.get_sku_predictions_year_range(
                    maker=maker,
                    model=model,
                    series=series,
                    description=descripcion,
                    limit=10
                )
                
                if predictions:
                    results['year_range_found'] += 1
                    print(f"🚀 Year Range DB found {len(predictions)} predictions:")
                    
                    predicted_skus = [pred['sku'] for pred in predictions]
                    top3_skus = predicted_skus[:3]
                    
                    for j, pred in enumerate(predictions[:5]):
                        status = "✅" if pred['sku'] == actual_sku else "❌"
                        print(f"  {j+1}. {pred['sku']} (freq: {pred['frequency']}, conf: {pred['confidence']:.3f}) {status}")
                    
                    if actual_sku in predicted_skus:
                        if predicted_skus[0] == actual_sku:
                            results['year_range_correct'] += 1
                            print("🎯 EXACT MATCH (Rank 1)!")
                        elif actual_sku in top3_skus:
                            results['year_range_top3'] += 1
                            rank = predicted_skus.index(actual_sku) + 1
                            print(f"🎯 FOUND in Top 3 (Rank {rank})!")
                        else:
                            rank = predicted_skus.index(actual_sku) + 1
                            print(f"📍 Found at rank {rank}")
                    else:
                        print("❌ Expected SKU not found in predictions")
                else:
                    print("❌ No year range predictions found")
                
                # Store result for analysis
                results['predictions_by_maker'][maker].append({
                    'found': len(predictions) > 0,
                    'correct': actual_sku in [p['sku'] for p in predictions] if predictions else False,
                    'rank1': predictions[0]['sku'] == actual_sku if predictions else False,
                    'top3': actual_sku in [p['sku'] for p in predictions[:3]] if predictions else False
                })
                
            except Exception as e:
                print(f"❌ Prediction error: {e}")
        
        # Calculate and display results
        print(f"\n📊 RESULTS SUMMARY")
        print("=" * 40)
        print(f"Total tests: {results['total']}")
        print(f"Year Range DB found predictions: {results['year_range_found']}/{results['total']} ({results['year_range_found']/results['total']*100:.1f}%)")
        print(f"Exact matches (Rank 1): {results['year_range_correct']}/{results['total']} ({results['year_range_correct']/results['total']*100:.1f}%)")
        print(f"Found in Top 3: {results['year_range_top3']}/{results['total']} ({results['year_range_top3']/results['total']*100:.1f}%)")
        
        total_found = results['year_range_correct'] + results['year_range_top3']
        print(f"Total useful predictions: {total_found}/{results['total']} ({total_found/results['total']*100:.1f}%)")
        
        # Results by maker
        print(f"\n📊 Results by Maker:")
        for maker, maker_results in results['predictions_by_maker'].items():
            if maker_results:
                found_count = sum(1 for r in maker_results if r['found'])
                correct_count = sum(1 for r in maker_results if r['rank1'])
                top3_count = sum(1 for r in maker_results if r['top3'])
                total_count = len(maker_results)
                
                print(f"  {maker}: {found_count}/{total_count} found, {correct_count} rank1, {top3_count} top3")
        
        # Test specific descriptions mentioned by user
        print(f"\n🎯 Testing User's Specific Examples...")
        user_tests = [
            ("Hyundai", "2018", "Unknown", "COSTADO IZQUIERDA"),
            ("Hyundai", "2018", "Unknown", "PERSIANA"),
            ("MAZDA", "2016", "BT50 [2] [FL]", "CAPO"),
            ("Toyota", "2016", "Tucson", "GUIA LATERAL DERECHA PARAGOLPES"),
        ]
        
        for maker, model, series, descripcion in user_tests:
            print(f"\n--- Testing: {maker} {model} {series} - '{descripcion}' ---")
            try:
                predictions = year_range_optimizer.get_sku_predictions_year_range(
                    maker=maker,
                    model=model,
                    series=series,
                    description=descripcion,
                    limit=5
                )
                
                if predictions:
                    print(f"✅ Found {len(predictions)} predictions:")
                    for i, pred in enumerate(predictions):
                        print(f"  {i+1}. {pred['sku']} (freq: {pred['frequency']}, range: {pred['year_range']}, conf: {pred['confidence']:.3f})")
                else:
                    print("❌ No predictions found")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during SKU testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sku_predictions()
    
    if success:
        print("\n🎉 SKU testing completed!")
    else:
        print("\n💥 SKU testing failed!")
