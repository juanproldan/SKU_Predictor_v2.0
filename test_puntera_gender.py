#!/usr/bin/env python3
"""
Test script to verify gender agreement for PUNTERA IZQUIERDA PARAGOLPES TRASERO
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.text_utils import get_noun_gender, find_immediate_noun_for_adjective, expand_gender_dependent_abbreviation

def test_puntera_gender():
    """Test that PUNTERA is correctly identified as feminine"""
    print("=== Testing PUNTERA Gender Detection ===")
    
    # Test direct gender detection
    gender = get_noun_gender('puntera')
    print(f"get_noun_gender('puntera') = '{gender}'")
    assert gender == 'feminine', f"Expected 'feminine', got '{gender}'"
    
    # Test phrase: "PUNTERA IZQUIERDA PARAGOLPES TRASERO"
    words = ['PUNTERA', 'IZQUIERDA', 'PARAGOLPES', 'TRASERO']
    words_lower = [w.lower() for w in words]
    
    print(f"\nTesting phrase: {' '.join(words)}")
    
    # Test immediate noun detection for IZQUIERDA (position 1)
    immediate_noun = find_immediate_noun_for_adjective(words_lower, 1)
    print(f"find_immediate_noun_for_adjective(words, 1) = '{immediate_noun}'")
    assert immediate_noun == 'puntera', f"Expected 'puntera', got '{immediate_noun}'"
    
    # Test immediate noun detection for TRASERO (position 3)
    immediate_noun_trasero = find_immediate_noun_for_adjective(words_lower, 3)
    print(f"find_immediate_noun_for_adjective(words, 3) = '{immediate_noun_trasero}'")
    assert immediate_noun_trasero == 'paragolpes', f"Expected 'paragolpes', got '{immediate_noun_trasero}'"
    
    # Test gender of PARAGOLPES
    paragolpes_gender = get_noun_gender('paragolpes')
    print(f"get_noun_gender('paragolpes') = '{paragolpes_gender}'")
    assert paragolpes_gender == 'masculine', f"Expected 'masculine', got '{paragolpes_gender}'"
    
    print("\n=== Testing Abbreviation Expansion ===")
    
    # Test abbreviation expansion with gender agreement
    # IZQUIERDA should remain IZQUIERDA (feminine) because it modifies PUNTERA
    result_izq = expand_gender_dependent_abbreviation('izquierda', 'puntera', words_lower, 1)
    print(f"expand_gender_dependent_abbreviation('izquierda', 'puntera', words, 1) = '{result_izq}'")
    assert result_izq == 'izquierda', f"Expected 'izquierda', got '{result_izq}'"

    # TRASERO should remain TRASERO (masculine) because it modifies PARAGOLPES
    result_tras = expand_gender_dependent_abbreviation('trasero', 'paragolpes', words_lower, 3)
    print(f"expand_gender_dependent_abbreviation('trasero', 'paragolpes', words, 3) = '{result_tras}'")
    assert result_tras == 'trasero', f"Expected 'trasero', got '{result_tras}'"
    
    print("\n✅ All tests passed! Gender agreement is working correctly.")
    print("Expected result: PUNTERA IZQUIERDA PARAGOLPES TRASERO (no changes)")

if __name__ == '__main__':
    test_puntera_gender()
