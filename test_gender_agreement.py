#!/usr/bin/env python3
"""
Test script for improved Spanish gender agreement in automotive part descriptions.
"""

import sys
import os
sys.path.append('src')

from utils.text_utils import normalize_text, expand_linguistic_variations_text

def test_gender_agreement():
    """Test the improved gender agreement system."""

    print("=== TESTING IMPROVED SPANISH GENDER AGREEMENT ===\n")
    
    test_cases = [
        # The user's challenging test cases - ALL should work now!
        ("GUARDAPOLVO PLASTICO D IZQU", "guardapolvo plastico delantero izquierdo"),
        ("GUARDAPOLVO PLASTICO D I", "guardapolvo plastico delantero izquierdo"),
        ("GUARDAPOLVO PLASTICO D IZ", "guardapolvo plastico delantero izquierdo"),
        ("GUARDAPOLVO PLASTICO DEL IZQU", "guardapolvo plastico delantero izquierdo"),

        # More comprehensive abbreviation combinations
        ("GUARDAPOLVO PLASTICO DE IZ", "guardapolvo plastico delantero izquierdo"),
        ("GUARDAPOLVO PLASTICO DEL I", "guardapolvo plastico delantero izquierdo"),
        ("GUARDAPOLVO PLASTICO D DER", "guardapolvo plastico delantero derecho"),
        ("GUARDAPOLVO PLASTICO DEL DERECH", "guardapolvo plastico delantero derecho"),

        # Feminine noun cases with all combinations
        ("GUIA D IZQU", "guia delantera izquierda"),
        ("GUIA DEL IZ", "guia delantera izquierda"),
        ("FAROLA T DER", "farola trasera derecha"),
        ("FAROLA TRA DERECH", "farola trasera derecha"),

        # Rear position abbreviations
        ("FARO T I", "faro trasero izquierdo"),
        ("FARO TRA IZQU", "faro trasero izquierdo"),
        ("ESPEJO T DER", "espejo trasero derecho"),
        ("LUZ TRAS IZ", "luz trasera izquierda"),

        # Original test cases
        ("GUIA LATERAL IZ PARAGOLPES DEL", "guia lateral izquierda paragolpes delantera"),
        ("FAROLA DEL DER", "farola delantera derecha"),
        ("LUZ TRA IZ", "luz trasera izquierda"),
        ("GUARDAFANGO DEL DER", "guardafango delantero derecho"),
        ("PARAGOLPES TRA IZ", "paragolpes trasero izquierdo"),
    ]

    for original, expected in test_cases:
        # Test with full normalization
        result_full = normalize_text(original, expand_linguistic_variations=True)
        
        # Test with just linguistic variations
        result_linguistic = expand_linguistic_variations_text(original.lower())
        
        print(f"Original: '{original}'")
        print(f"Expected: '{expected}'")
        print(f"Full norm: '{result_full}'")
        print(f"Linguistic: '{result_linguistic}'")
        
        # Check if the result matches expected
        if result_full == expected:
            print("✅ PASS - Full normalization correct")
        else:
            print("❌ FAIL - Full normalization incorrect")
            
        if result_linguistic == expected:
            print("✅ PASS - Linguistic expansion correct")
        else:
            print("❌ FAIL - Linguistic expansion incorrect")
        
        print("-" * 60)

if __name__ == "__main__":
    test_gender_agreement()
