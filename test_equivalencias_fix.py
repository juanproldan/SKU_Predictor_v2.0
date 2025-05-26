#!/usr/bin/env python3
"""
Test script to verify the Equivalencias synonym expansion fix.

This script simulates the part processing logic to demonstrate that
synonym variations now produce consistent results.
"""

import sys
import os

# Add src to path
sys.path.append('src')

# Import the main app to access the global variables and methods
from main_app import FixacarApp
import tkinter as tk

def test_synonym_expansion_fix():
    """Test that synonym expansion produces consistent results."""
    print("🧪 TESTING EQUIVALENCIAS SYNONYM EXPANSION FIX")
    print("=" * 60)
    
    # Create a minimal tkinter root (hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    try:
        # Create app instance to load data
        print("📊 Loading application data...")
        app = FixacarApp(root)
        
        # Test inputs - these should now produce consistent results
        test_cases = [
            "FAROLA IZQUIERDA",  # Original (should work)
            "FAROLA IZQ",        # Synonym (should now work)
            "FAROLA IZ",         # Synonym (should now work)
        ]
        
        print(f"\n🔍 Testing synonym expansion for {len(test_cases)} cases:")
        print("-" * 60)
        
        results = []
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: '{test_input}'")
            
            # Apply the same processing as the main app
            print(f"   Step 1: Original input: '{test_input}'")
            
            # Apply synonym expansion
            expanded = app.expand_synonyms(test_input)
            print(f"   Step 2: After expansion: '{expanded}'")
            
            # Apply normalization
            from utils.text_utils import normalize_text
            normalized = normalize_text(expanded)
            print(f"   Step 3: After normalization: '{normalized}'")
            
            # Check equivalencia ID
            from main_app import equivalencias_map_global
            eq_id = equivalencias_map_global.get(normalized)
            print(f"   Step 4: Equivalencia ID: {eq_id}")
            
            # Store result
            results.append({
                'input': test_input,
                'expanded': expanded,
                'normalized': normalized,
                'eq_id': eq_id,
                'consistent': expanded == results[0]['expanded'] if results else True
            })
            
            print(f"   ✅ Result: '{test_input}' -> '{normalized}' (EqID: {eq_id})")
        
        # Analyze results
        print("\n📋 RESULTS ANALYSIS:")
        print("=" * 60)
        
        # Check if all variations produce the same expanded form
        expanded_forms = [r['expanded'] for r in results]
        normalized_forms = [r['normalized'] for r in results]
        eq_ids = [r['eq_id'] for r in results]
        
        all_expanded_same = len(set(expanded_forms)) == 1
        all_normalized_same = len(set(normalized_forms)) == 1
        all_eq_ids_same = len(set(eq_ids)) == 1
        
        print(f"🎯 Expanded forms consistent: {'✅ YES' if all_expanded_same else '❌ NO'}")
        print(f"🎯 Normalized forms consistent: {'✅ YES' if all_normalized_same else '❌ NO'}")
        print(f"🎯 Equivalencia IDs consistent: {'✅ YES' if all_eq_ids_same else '❌ NO'}")
        
        if all_expanded_same and all_normalized_same and all_eq_ids_same:
            print(f"\n🎉 SUCCESS! All synonym variations now produce consistent results:")
            print(f"   📝 All inputs expand to: '{expanded_forms[0]}'")
            print(f"   📝 All inputs normalize to: '{normalized_forms[0]}'")
            print(f"   📝 All inputs get EqID: {eq_ids[0]}")
            print(f"\n✅ EQUIVALENCIAS SYNONYM EXPANSION FIX: WORKING CORRECTLY")
        else:
            print(f"\n❌ ISSUE: Synonym variations still produce different results")
            for r in results:
                print(f"   '{r['input']}' -> '{r['expanded']}' -> '{r['normalized']}' (EqID: {r['eq_id']})")
        
        print("\n" + "=" * 60)
        print("🏁 TEST COMPLETE")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        root.destroy()

if __name__ == "__main__":
    test_synonym_expansion_fix()
