#!/usr/bin/env python3
"""
Test script to verify the Unified Text Preprocessing Pipeline
for eliminating false penalties in SKU prediction fuzzy matching.

This script tests that linguistically equivalent terms like:
- "FAROLA IZQ" vs "FAROLA IZQUIERDA" 
- "FARO DER" vs "FARO DELANTERO DERECHO"
- "GUARDAPOLVO" vs "GUARDAPOLVO PLASTICO"

Are properly normalized to identical strings, achieving perfect matches (1.0 similarity)
instead of penalized fuzzy matches (0.85).
"""

import sys
import os
import tkinter as tk

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main_app import FixacarApp
from utils.text_utils import normalize_text

def test_unified_preprocessing():
    """Test the unified text preprocessing pipeline"""

    print("🔬 Testing Unified Text Preprocessing Pipeline")
    print("=" * 60)

    # Create a root window for the app (required for tkinter)
    root = tk.Tk()
    root.withdraw()  # Hide the window since we're just testing

    # Create app instance to access unified preprocessing
    app = FixacarApp(root)
    
    # Test cases: linguistically equivalent terms that should produce identical results
    test_cases = [
        # Abbreviation variations
        ("FAROLA IZQ", "FAROLA IZQUIERDA"),
        ("FARO DER", "FARO DERECHO"),
        ("GUARDAPOLVO PLAST", "GUARDAPOLVO PLASTICO"),
        
        # Gender agreement variations
        ("BOCEL IZQUIERDO", "BOCEL IZQUIERDA"),
        ("REMACHE NEGRO", "REMACHE NEGRA"),
        
        # Case variations
        ("farola izquierda", "FAROLA IZQUIERDA"),
        ("Faro Delantero", "FARO DELANTERO"),
        
        # Plural/singular variations
        ("CALCOMANIA", "CALCOMANIAS"),
        ("REMACHE", "REMACHES"),
        
        # Complex combinations
        ("FAROLA IZQ CROMADO", "FARO DELANTERO IZQUIERDA CROMADA"),
        ("GUARDAPOLVO PLAST DER", "GUARDAPOLVO PLASTICO DERECHO"),
    ]
    
    print("\n📋 Test Results:")
    print("-" * 60)
    
    perfect_matches = 0
    total_tests = len(test_cases)
    
    for i, (input1, input2) in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: Comparing linguistically equivalent terms")
        print(f"   Input 1: '{input1}'")
        print(f"   Input 2: '{input2}'")
        
        # Apply unified preprocessing to both inputs
        processed1 = app.unified_text_preprocessing(input1)
        processed2 = app.unified_text_preprocessing(input2)
        
        print(f"   Processed 1: '{processed1}'")
        print(f"   Processed 2: '{processed2}'")
        
        # Check if they match perfectly after preprocessing
        is_perfect_match = processed1 == processed2
        
        if is_perfect_match:
            print(f"   ✅ PERFECT MATCH: Identical after preprocessing")
            perfect_matches += 1
        else:
            print(f"   ❌ MISMATCH: Different after preprocessing")
            
            # Calculate similarity for comparison
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, processed1, processed2).ratio()
            print(f"   📊 Similarity: {similarity:.3f}")
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY: {perfect_matches}/{total_tests} tests achieved perfect matches")
    
    success_rate = (perfect_matches / total_tests) * 100
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("✅ EXCELLENT: Unified preprocessing is working effectively!")
    elif success_rate >= 60:
        print("⚠️  GOOD: Most cases handled, some edge cases remain")
    else:
        print("❌ NEEDS IMPROVEMENT: Many linguistic equivalents still mismatched")
    
    return success_rate

def test_synonym_expansion():
    """Test industry synonym expansion from Equivalencias.xlsx"""

    print("\n\n🔬 Testing Industry Synonym Expansion")
    print("=" * 60)

    # Create a root window for the app (required for tkinter)
    root = tk.Tk()
    root.withdraw()  # Hide the window since we're just testing

    app = FixacarApp(root)
    
    # Test cases for industry synonyms (these depend on Equivalencias.xlsx content)
    synonym_test_cases = [
        "FAROLA",  # Should expand to "FARO" or similar
        "GUARDAPOLVO",  # Should expand to industry standard term
        "BOCEL",  # Should expand to industry standard term
        "CALCOMANIA",  # Should expand to industry standard term
    ]
    
    print("\n📋 Synonym Expansion Results:")
    print("-" * 60)
    
    for term in synonym_test_cases:
        print(f"\n🔍 Testing: '{term}'")
        
        # Test original expand_synonyms method
        expanded = app.expand_synonyms(term)
        print(f"   Expanded: '{expanded}'")
        
        # Test unified preprocessing (includes synonym expansion)
        unified = app.unified_text_preprocessing(term)
        print(f"   Unified: '{unified}'")
        
        if expanded != term:
            print(f"   ✅ Synonym expansion applied")
        else:
            print(f"   ℹ️  No synonym found (may be expected)")

def test_real_world_scenarios():
    """Test real-world scenarios from the VIN prediction example"""

    print("\n\n🔬 Testing Real-World VIN Prediction Scenarios")
    print("=" * 60)

    # Create a root window for the app (required for tkinter)
    root = tk.Tk()
    root.withdraw()  # Hide the window since we're just testing

    app = FixacarApp(root)
    
    # Real examples from the user's VIN prediction
    real_world_cases = [
        # User input vs Database variations
        ("FAROLA IZQUIERDA", "FAROLA IZQ"),
        ("GUARDAPOLVO PLASTICO DELANTERO", "GUARDAPOLVO PLAST DER"),
        ("BOCEL IZQUIERDO PERSIANA CROMADO", "BOCEL IZQ PERSIANA CROMADA"),
        ("CALCOMANIA CAPO", "CALCOMANIAS CAPO"),
        ("GUIA LATERAL IZQUIERDA PARAGOLPES", "GUIA LAT IZQ PARAGOLPES"),
        ("REMACHE NEGRO DE PERSIANA", "REMACHE NEGRA PERSIANA"),
    ]
    
    print("\n📋 Real-World Test Results:")
    print("-" * 60)
    
    for i, (user_input, db_variant) in enumerate(real_world_cases, 1):
        print(f"\n🎯 Real-World Test {i}:")
        print(f"   User Input: '{user_input}'")
        print(f"   DB Variant: '{db_variant}'")
        
        # Apply unified preprocessing
        processed_input = app.unified_text_preprocessing(user_input)
        processed_db = app.unified_text_preprocessing(db_variant)
        
        print(f"   Processed Input: '{processed_input}'")
        print(f"   Processed DB: '{processed_db}'")
        
        # Check match
        is_match = processed_input == processed_db
        
        if is_match:
            print(f"   ✅ PERFECT MATCH: Will get 1.0 confidence instead of fuzzy penalty")
        else:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, processed_input, processed_db).ratio()
            print(f"   📊 Similarity: {similarity:.3f} (still fuzzy match)")
            
            if similarity >= 0.9:
                print(f"   ⚠️  VERY CLOSE: Minor differences remain")
            else:
                print(f"   ❌ SIGNIFICANT DIFFERENCE: Needs improvement")

if __name__ == "__main__":
    print("🚀 Starting Unified Text Preprocessing Tests")
    print("=" * 80)
    
    try:
        # Run all tests
        success_rate = test_unified_preprocessing()
        test_synonym_expansion()
        test_real_world_scenarios()
        
        print("\n" + "=" * 80)
        print("🏁 Testing Complete!")
        
        if success_rate >= 80:
            print("🎉 The unified preprocessing pipeline is working excellently!")
            print("   Linguistically equivalent terms should now achieve perfect matches.")
        else:
            print("⚠️  The preprocessing pipeline needs further refinement.")
            print("   Some linguistically equivalent terms are still being penalized.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
