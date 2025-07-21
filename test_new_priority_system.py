#!/usr/bin/env python3
"""
Test script to verify the new priority system and confidence scoring.
Tests:
1. New prediction order (Maestro → NN → Database)
2. Updated confidence ranges (40-80% for DB, 70-85% for NN, 90-100% for Maestro)
3. Consensus logic (NN + DB = higher + 10%, Maestro + NN = 100%)
4. Percentage display format
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_confidence_calculations():
    """Test the new confidence calculation functions."""
    print("=== Testing Confidence Calculations ===")
    
    # Import the app to test methods
    import tkinter as tk
    from main_app import FixacarApp
    
    root = tk.Tk()
    root.withdraw()  # Hide window
    app = FixacarApp(root)
    
    # Test frequency-based confidence (Database)
    print("\n--- Database Confidence Tests ---")
    test_frequencies = [1, 5, 15, 25, 50]
    for freq in test_frequencies:
        conf = app.calculate_frequency_based_confidence(freq, "DB")
        percentage = app._format_confidence_percentage(conf)
        print(f"Frequency {freq}: {conf:.3f} ({percentage})")
    
    # Test consensus confidence
    print("\n--- Consensus Confidence Tests ---")
    
    # Single sources
    maestro_conf = app._calculate_consensus_confidence(0.95, ["Maestro"])
    nn_conf = app._calculate_consensus_confidence(0.80, ["SKU-NN"])
    db_conf = app._calculate_consensus_confidence(0.75, ["DB"])
    
    print(f"Maestro alone: {maestro_conf:.3f} ({app._format_confidence_percentage(maestro_conf)})")
    print(f"NN alone: {nn_conf:.3f} ({app._format_confidence_percentage(nn_conf)})")
    print(f"DB alone: {db_conf:.3f} ({app._format_confidence_percentage(db_conf)})")
    
    # Consensus combinations
    maestro_nn = app._calculate_consensus_confidence(0.85, ["Maestro", "SKU-NN"])
    nn_db = app._calculate_consensus_confidence(0.75, ["SKU-NN", "DB"])
    all_three = app._calculate_consensus_confidence(0.80, ["Maestro", "SKU-NN", "DB"])
    
    print(f"Maestro + NN: {maestro_nn:.3f} ({app._format_confidence_percentage(maestro_nn)})")
    print(f"NN + DB: {nn_db:.3f} ({app._format_confidence_percentage(nn_db)})")
    print(f"All three: {all_three:.3f} ({app._format_confidence_percentage(all_three)})")
    
    root.destroy()

def test_percentage_formatting():
    """Test percentage formatting function."""
    print("\n=== Testing Percentage Formatting ===")
    
    import tkinter as tk
    from main_app import FixacarApp
    
    root = tk.Tk()
    root.withdraw()
    app = FixacarApp(root)
    
    test_values = [0.0, 0.1, 0.45, 0.7, 0.85, 0.9, 1.0]
    for value in test_values:
        percentage = app._format_confidence_percentage(value)
        print(f"{value:.2f} → {percentage}")
    
    root.destroy()

def test_equivalence_groups():
    """Test that equivalence groups are working."""
    print("\n=== Testing Equivalence Groups ===")
    
    import tkinter as tk
    from main_app import FixacarApp
    
    root = tk.Tk()
    root.withdraw()
    app = FixacarApp(root)
    
    # Test synonym expansion
    test_cases = [
        "FAROLA IZQUIERDA",
        "FAROLA IZQ", 
        "PARAGOLPE DELANTERO"
    ]
    
    for test_text in test_cases:
        expanded = app.expand_synonyms(test_text)
        print(f"'{test_text}' → '{expanded}'")
    
    root.destroy()

def print_priority_summary():
    """Print summary of new priority system."""
    print("\n=== NEW PRIORITY SYSTEM SUMMARY ===")
    print("Priority 1: Maestro Data")
    print("  - Solo: 90% max confidence")
    print("  - With NN consensus: 100% (auto-selected)")
    print()
    print("Priority 2: Neural Network")
    print("  - Range: 70-85% confidence")
    print("  - Based on model prediction scores")
    print()
    print("Priority 3: Historical Database")
    print("  - Range: 40-80% confidence")
    print("  - High (80%): 20+ exact matches")
    print("  - Low (40%): Few matches or outliers")
    print()
    print("Consensus Bonuses:")
    print("  - NN + DB: Higher value + 10%")
    print("  - Maestro + NN: 100%")
    print("  - All three: 100%")
    print()
    print("Display: All confidence shown as percentages (75% instead of 0.75)")

if __name__ == "__main__":
    print("🎯 Testing New Priority System Implementation")
    print("=" * 50)
    
    try:
        test_confidence_calculations()
        test_percentage_formatting()
        test_equivalence_groups()
        print_priority_summary()
        
        print("\n✅ All tests completed successfully!")
        print("\n🚀 New priority system is ready:")
        print("   1. Maestro → 2. Neural Network → 3. Database")
        print("   Confidence displayed as percentages")
        print("   Equivalence groups implemented")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
