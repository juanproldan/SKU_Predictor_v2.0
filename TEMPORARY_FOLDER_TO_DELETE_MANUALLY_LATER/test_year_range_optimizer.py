#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the year range database optimizer
"""

import os
import sys

# Add src to path
sys.path.append('src')

from utils.year_range_database import YearRangeDatabaseOptimizer

def get_base_path():
    """Get the base path for the application."""
    return os.path.join(os.getcwd(), "Fixacar_SKU_Predictor_CLIENT")

def test_year_range_optimizer():
    """Test the year range database optimizer."""
    
    db_path = os.path.join(get_base_path(), "Source_Files", "processed_consolidado.db")
    
    print(f"Testing year range optimizer with database: {db_path}")
    
    try:
        # Initialize optimizer
        optimizer = YearRangeDatabaseOptimizer(db_path)
        print("✅ Year range optimizer initialized")
        
        # Get statistics
        stats = optimizer.get_year_range_statistics()
        print(f"✅ Statistics retrieved: {stats}")
        
        # Test SKU prediction
        print("\n=== Testing SKU Prediction ===")
        predictions = optimizer.get_sku_predictions_year_range(
            maker="Toyota",
            model=2020,
            series="Corolla", 
            description="test part",
            limit=5
        )
        
        print(f"SKU predictions found: {len(predictions)}")
        for pred in predictions:
            print(f"  - SKU: {pred['sku']}, Confidence: {pred['confidence']:.3f}, Source: {pred['source']}, Range: {pred['year_range']}")
        
        # Test VIN prediction
        print("\n=== Testing VIN Prediction ===")
        vin_predictions = optimizer.get_vin_predictions_year_range(
            maker="Toyota",
            series="Corolla",
            model=2020
        )
        
        print(f"VIN predictions found: {len(vin_predictions)}")
        for pred in vin_predictions:
            print(f"  - Confidence: {pred['confidence']:.3f}, Source: {pred['source']}, Range: {pred['year_range']}")
        
        # Test with different year (should still match range)
        print("\n=== Testing Year Range Matching ===")
        predictions_2019 = optimizer.get_sku_predictions_year_range(
            maker="Toyota",
            model=2019,  # Different year, but should still match if in range
            series="Corolla", 
            description="test part",
            limit=5
        )
        
        print(f"SKU predictions for 2019 (should be empty since range is 2020-2020): {len(predictions_2019)}")
        
        # Test with year in range
        predictions_2020 = optimizer.get_sku_predictions_year_range(
            maker="Toyota",
            model=2020,  # Exact year match
            series="Corolla", 
            description="test part",
            limit=5
        )
        
        print(f"SKU predictions for 2020 (should match): {len(predictions_2020)}")
        
        optimizer.close()
        print("✅ Year range optimizer test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Year range optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_year_range_optimizer()
    if success:
        print("\n🎉 Year range optimizer test passed!")
    else:
        print("\n💥 Year range optimizer test failed!")
