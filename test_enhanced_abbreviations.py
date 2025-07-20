#!/usr/bin/env python3
"""
Test script for enhanced abbreviation and linguistic variation system.
Tests case insensitivity, abbreviations, gender variations, and plurals.
"""

import sys
import os
sys.path.append('src')

from utils.text_utils import normalize_text, expand_linguistic_variations_text

def test_linguistic_variations():
    """Test the enhanced linguistic variation system."""

    print("=== TESTING ENHANCED LINGUISTIC VARIATION SYSTEM ===\n")
    
    print("=== TEST 1: CASE INSENSITIVITY ===")
    test_cases_case = [
        "FAROLA IZQUIERDA",
        "farola izquierda",
        "Farola Izquierda",
        "FaRoLa IzQuIeRdA"
    ]

    for test_case in test_cases_case:
        normalized = normalize_text(test_case)
        print(f"'{test_case}' -> '{normalized}'")

    print("\n=== TEST 2: CONTEXT-DEPENDENT ABBREVIATIONS ===")
    test_cases_context = [
        # LATERAL PARTS (d = derecha, i = izquierda)
        "farola d",         # Should become 'farola derecha'
        "farola i",         # Should become 'farola izquierda'
        "espejo d",         # Should become 'espejo derecha'
        "guardafango d",    # Should become 'guardafango derecha'

        # LONGITUDINAL PARTS (d = delantero, t = trasero)
        "paragolpes d",     # Should become 'paragolpes delantero'
        "paragolpes t",     # Should become 'paragolpes trasero'
        "traviesa d",       # Should become 'traviesa delantero'
        "traviesa t",       # Should become 'traviesa trasero'
    ]

    for test_case in test_cases_context:
        normalized = normalize_text(test_case)
        print(f"'{test_case}' -> '{normalized}'")

    print("\n=== TEST 2b: STANDARD ABBREVIATIONS ===")
    test_cases_abbr = [
        "farola iz",
        "farola izq",
        "farola der",
        "farola dere",
        "farola del",
        "farola delan",
        "farola tra",
        "farola tras"
    ]

    for test_case in test_cases_abbr:
        normalized = normalize_text(test_case)
        print(f"'{test_case}' -> '{normalized}'")

    print("\n=== TEST 3: GENDER VARIATIONS (Should NOT be normalized) ===")
    test_cases_gender = [
        "guardafango derecho",
        "guardafango derecha",
        "paragolpes delantero",
        "paragolpes delantera",
        "espejo izquierdo",
        "espejo izquierda"
    ]

    for test_case in test_cases_gender:
        normalized = normalize_text(test_case)
        print(f"'{test_case}' -> '{normalized}' (should preserve gender)")

    print("\n=== TEST 3b: GENDER SIMILARITY TESTING ===")
    from utils.text_utils import are_gender_variants
    gender_test_pairs = [
        ("derecho", "derecha"),
        ("izquierdo", "izquierda"),
        ("delantero", "delantera"),
        ("trasero", "trasera"),
        ("farola", "farola"),  # Same word - should be False
    ]

    for word1, word2 in gender_test_pairs:
        is_variant = are_gender_variants(word1, word2)
        print(f"'{word1}' vs '{word2}' -> Gender variants: {is_variant}")

    print("\n=== TEST 4: PLURAL VARIATIONS ===")
    test_cases_plural = [
        "farola",
        "farolas",
        "luz",
        "luces",
        "espejo",
        "espejos",
        "paragolpe",
        "paragolpes"
    ]

    for test_case in test_cases_plural:
        normalized = normalize_text(test_case)
        print(f"'{test_case}' -> '{normalized}'")

    print("\n=== TEST 5: COMPLEX COMBINATIONS ===")
    test_cases_complex = [
        "FAROLA IZQ",
        "luz antiniebla der",
        "GUARDAFANGO I",
        "paragolpes del",
        "Espejo D",
        "TRAVIESA SUP"
    ]

    for test_case in test_cases_complex:
        normalized = normalize_text(test_case)
        print(f"'{test_case}' -> '{normalized}'")

    print("\n=== TEST 6: MAKE NAMES (CASE INSENSITIVITY) ===")
    test_cases_makes = [
        "RENAULT",
        "renault",
        "Renault",
        "ReNaUlT"
    ]

    for test_case in test_cases_makes:
        normalized = normalize_text(test_case)
        print(f"'{test_case}' -> '{normalized}'")

if __name__ == "__main__":
    test_linguistic_variations()
