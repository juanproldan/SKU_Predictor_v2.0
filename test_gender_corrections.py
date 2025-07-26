#!/usr/bin/env python3
"""
Test script to verify gender agreement corrections in the text processing pipeline
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.text_utils import normalize_text

def test_gender_corrections():
    """Test various gender agreement scenarios"""
    print("=== Testing Gender Agreement Corrections ===")
    
    test_cases = [
        {
            'input': 'PUNTERA IZQUIERDA PARAGOLPES TRASERO',
            'expected': 'puntera izquierda paragolpes trasero',
            'description': 'PUNTERA (feminine) + IZQUIERDA (should stay feminine), PARAGOLPES (masculine) + TRASERO (should stay masculine)'
        },
        {
            'input': 'PUNTERA IZQUIERDO PARAGOLPES TRASERO',
            'expected': 'puntera izquierda paragolpes trasero',
            'description': 'PUNTERA (feminine) + IZQUIERDO (should become IZQUIERDA)'
        },
        {
            'input': 'GUIA LATERAL IZQUIERDA PARAGOLPES DELANTERO',
            'expected': 'guia lateral izquierda paragolpes delantero',
            'description': 'GUIA (feminine) + LATERAL IZQUIERDA (should stay feminine), PARAGOLPES (masculine) + DELANTERO (should stay masculine)'
        },
        {
            'input': 'GUARDAPOLVO DELANTERO DERECHO',
            'expected': 'guardapolvo delantero derecho',
            'description': 'GUARDAPOLVO (masculine) + DELANTERO DERECHO (should stay masculine)'
        },
        {
            'input': 'PUERTA DELANTERA DERECHA',
            'expected': 'puerta delantera derecha',
            'description': 'PUERTA (feminine) + DELANTERA DERECHA (should stay feminine)'
        },
        {
            'input': 'PUERTA DELANTERO DERECHO',
            'expected': 'puerta delantera derecha',
            'description': 'PUERTA (feminine) + DELANTERO DERECHO (should become DELANTERA DERECHA)'
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['description']} ---")
        print(f"Input:    '{test_case['input']}'")
        
        result = normalize_text(test_case['input'], expand_linguistic_variations=True)
        print(f"Result:   '{result}'")
        print(f"Expected: '{test_case['expected']}'")
        
        if result == test_case['expected']:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("✅ All gender agreement tests passed!")
    else:
        print("❌ Some tests failed. Gender agreement needs fixing.")
    
    return all_passed

if __name__ == '__main__':
    test_gender_corrections()
