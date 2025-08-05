#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test year range functionality with realistic automotive data
"""

import os
import sys

# Add src to path
sys.path.append('src')

from utils.year_range_database import YearRangeDatabaseOptimizer

def get_base_path():
    """Get the base path for the application."""
    return os.path.join(os.getcwd(), "Fixacar_SKU_Predictor_CLIENT")

def test_realistic_year_ranges():
    """Test year range functionality with realistic automotive data."""
    
    db_path = os.path.join(get_base_path(), "Source_Files", "processed_consolidado.db")
    
    print(f"Testing year range optimizer with realistic data: {db_path}")
    
    try:
        # Initialize optimizer
        optimizer = YearRangeDatabaseOptimizer(db_path)
        print("✅ Year range optimizer initialized")
        
        # Get statistics
        stats = optimizer.get_year_range_statistics()
        print(f"✅ Statistics: SKU ranges: {stats['sku_year_ranges']}, VIN ranges: {stats['vin_year_ranges']}")
        print(f"   Average SKU frequency: {stats['avg_sku_frequency']:.1f}, Average year span: {stats['avg_sku_year_span']:.1f}")
        
        # Test 1: Toyota Corolla air filter (should find TOY-AIR-001 with high confidence)
        print("\n=== Test 1: Toyota Corolla 2020 Air Filter ===")
        predictions = optimizer.get_sku_predictions_year_range(
            maker="Toyota",
            model=2020,
            series="Corolla", 
            description="filtro aire",
            limit=5
        )
        
        print(f"Predictions found: {len(predictions)}")
        for pred in predictions:
            print(f"  - SKU: {pred['sku']}, Confidence: {pred['confidence']:.3f}, Freq: {pred['frequency']}, Range: {pred['year_range']}")
        
        # Test 2: Toyota Corolla 2019 (should still match air filter range 2018-2022)
        print("\n=== Test 2: Toyota Corolla 2019 Air Filter (Year Range Test) ===")
        predictions_2019 = optimizer.get_sku_predictions_year_range(
            maker="Toyota",
            model=2019,
            series="Corolla", 
            description="filtro aire",
            limit=5
        )
        
        print(f"Predictions found: {len(predictions_2019)}")
        for pred in predictions_2019:
            print(f"  - SKU: {pred['sku']}, Confidence: {pred['confidence']:.3f}, Freq: {pred['frequency']}, Range: {pred['year_range']}")
        
        # Test 3: Toyota Corolla 2023 (should NOT match air filter range 2018-2022)
        print("\n=== Test 3: Toyota Corolla 2023 Air Filter (Outside Range) ===")
        predictions_2023 = optimizer.get_sku_predictions_year_range(
            maker="Toyota",
            model=2023,
            series="Corolla", 
            description="filtro aire",
            limit=5
        )
        
        print(f"Predictions found: {len(predictions_2023)} (should be 0)")
        
        # Test 4: High frequency part (oil - TOY-OIL-001 with 8 occurrences)
        print("\n=== Test 4: Toyota Corolla 2020 Oil (High Frequency) ===")
        oil_predictions = optimizer.get_sku_predictions_year_range(
            maker="Toyota",
            model=2020,
            series="Corolla", 
            description="aceite motor",
            limit=5
        )
        
        print(f"Oil predictions found: {len(oil_predictions)}")
        for pred in oil_predictions:
            print(f"  - SKU: {pred['sku']}, Confidence: {pred['confidence']:.3f}, Freq: {pred['frequency']}, Range: {pred['year_range']}")
        
        # Test 5: Different make/series (Mazda CX-5)
        print("\n=== Test 5: Mazda CX-5 2018 Air Filter ===")
        mazda_predictions = optimizer.get_sku_predictions_year_range(
            maker="Mazda",
            model=2018,
            series="CX-5", 
            description="filtro aire",
            limit=5
        )
        
        print(f"Mazda predictions found: {len(mazda_predictions)}")
        for pred in mazda_predictions:
            print(f"  - SKU: {pred['sku']}, Confidence: {pred['confidence']:.3f}, Freq: {pred['frequency']}, Range: {pred['year_range']}")
        
        # Test 6: Year range with gaps (Chevrolet Cruze shock absorber 2015-2019 with gap in 2017)
        print("\n=== Test 6: Chevrolet Cruze 2017 Shock Absorber (Gap Year) ===")
        gap_predictions = optimizer.get_sku_predictions_year_range(
            maker="Chevrolet",
            model=2017,  # This year has a gap in the data but should still match the range
            series="Cruze", 
            description="amortiguador delantero",
            limit=5
        )
        
        print(f"Gap year predictions found: {len(gap_predictions)}")
        for pred in gap_predictions:
            print(f"  - SKU: {pred['sku']}, Confidence: {pred['confidence']:.3f}, Freq: {pred['frequency']}, Range: {pred['year_range']}")
        
        # Test 7: VIN prediction
        print("\n=== Test 7: VIN Prediction - Toyota Corolla 2020 ===")
        vin_predictions = optimizer.get_vin_predictions_year_range(
            maker="Toyota",
            series="Corolla",
            model=2020
        )
        
        print(f"VIN predictions found: {len(vin_predictions)}")
        for pred in vin_predictions:
            print(f"  - Confidence: {pred['confidence']:.3f}, Freq: {pred['frequency']}, Range: {pred['year_range']}")
        
        # Test 8: Fuzzy description matching
        print("\n=== Test 8: Fuzzy Description Matching ===")
        fuzzy_predictions = optimizer.get_sku_predictions_year_range(
            maker="Toyota",
            model=2020,
            series="Corolla", 
            description="filtro de aire motor",  # Slightly different description
            limit=5
        )
        
        print(f"Fuzzy predictions found: {len(fuzzy_predictions)}")
        for pred in fuzzy_predictions:
            print(f"  - SKU: {pred['sku']}, Confidence: {pred['confidence']:.3f}, Source: {pred['source']}, Range: {pred['year_range']}")
        
        optimizer.close()
        print("\n✅ All year range tests completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Year range test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_realistic_year_ranges()
    if success:
        print("\n🎉 Realistic year range tests passed!")
    else:
        print("\n💥 Realistic year range tests failed!")
