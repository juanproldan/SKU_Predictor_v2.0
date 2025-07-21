#!/usr/bin/env python3
"""
Test script to verify the equivalence groups implementation.
This script tests the new equivalence group system instead of canonical forms.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main_app import FixacarApp
import tkinter as tk

def test_equivalence_groups():
    """Test the equivalence groups functionality."""
    print("=== Testing Equivalence Groups Implementation ===")
    
    # Create a minimal tkinter root for testing
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    # Create app instance
    app = FixacarApp(root)
    
    # Test synonym expansion
    test_cases = [
        "FAROLA IZQUIERDA",
        "FAROLA IZQ", 
        "FAROLA IZ",
        "FARO IZQUIERDO",
        "PARAGOLPE DELANTERO",
        "PARAGOLPES DELANTERO"
    ]
    
    print("\n--- Testing Synonym Expansion ---")
    for test_text in test_cases:
        expanded = app.expand_synonyms(test_text)
        print(f"'{test_text}' -> '{expanded}'")
    
    # Test unified preprocessing
    print("\n--- Testing Unified Text Preprocessing ---")
    for test_text in test_cases:
        processed = app.unified_text_preprocessing(test_text)
        print(f"'{test_text}' -> '{processed}'")
    
    # Check if equivalence groups are loaded
    from main_app import synonym_expansion_map_global
    print(f"\n--- Equivalence Groups Loaded ---")
    print(f"Total synonym mappings: {len(synonym_expansion_map_global)}")
    
    if synonym_expansion_map_global:
        print("\nSample mappings:")
        for i, (term, group_id) in enumerate(list(synonym_expansion_map_global.items())[:5]):
            print(f"  '{term}' -> Group {group_id}")
    
    root.destroy()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_equivalence_groups()
