"""
Test script for the fuzzy matching functionality.

This script tests the fuzzy matching capabilities of the SKU Predictor application
by comparing the results of standard normalization and fuzzy normalization on a set
of test cases.
"""

from utils.fuzzy_matcher import (
    fuzzy_normalize_text,
    expand_abbreviations,
    split_compound_words,
    calculate_similarity,
    find_best_match,
    get_fuzzy_matches
)
from utils.text_utils import normalize_text
import sys
import os
from typing import Dict, List, Tuple

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the text normalization functions

# Test cases for normalization
TEST_CASES = [
    # Original text, Expected standard normalization, Expected fuzzy normalization
    ("izquierdo", "izquierdo", "izquierdo"),
    ("izq", "izq", "izquierdo"),
    ("iz", "iz", "izquierdo"),
    ("izquier", "izquier", "izquierdo"),
    ("i", "i", "izquierdo"),

    ("delantero", "delantero", "delantero"),
    ("del", "del", "delantero"),
    ("de", "de", "delantero"),
    ("d", "d", "derecho"),  # Note: 'd' is ambiguous, but we map it to 'derecho'
    ("delan", "delan", "delantero"),

    ("derecho", "derecho", "derecho"),
    ("der", "der", "derecho"),
    ("dere", "dere", "derecho"),

    ("trasero", "trasero", "trasero"),
    ("tra", "tra", "trasero"),
    ("t", "t", "trasero"),
    ("tras", "tras", "trasero"),

    # Compound words
    ("sopdparagolpes", "sopdparagolpes", "soporte de paragolpes"),
    ("guardbarroizq", "guardbarroizq", "guardabarro izquierdo"),
    ("paragolpesdelantero", "paragolpesdelantero", "paragolpes delantero"),
    ("farolaizquierda", "farolaizquierda", "farola izquierda"),

    # Mixed cases
    ("Guardabarro Izq", "guardabarro izq", "guardabarro izquierdo"),
    ("PARAGOLPES DEL", "paragolpes del", "paragolpes delantero"),
    ("Faro Tra. Der.", "faro tra der", "farola trasero derecho"),

    # Real-world examples
    ("soporte de paragolpes delantero izquierdo", "soporte de paragolpes delantero izquierdo",
     "soporte de paragolpes delantero izquierdo"),
    ("sop parag del izq", "sop parag del izq",
     "soporte paragolpes delantero izquierdo"),
    ("guardabarro delantero derecho", "guardabarro delantero derecho",
     "guardabarro delantero derecho"),
    ("guard del der", "guard del der", "guardabarro delantero derecho"),
]

# Test cases for fuzzy matching
FUZZY_MATCH_TESTS = [
    # Query, Candidates, Expected best match
    (
        "guardabarro izq",
        ["guardabarro derecho", "guardabarro izquierdo", "paragolpes delantero"],
        "guardabarro izquierdo"
    ),
    (
        "parag del",
        ["paragolpes trasero", "paragolpes delantero", "guardabarro delantero"],
        "paragolpes delantero"
    ),
    (
        "soporte parag",
        ["soporte motor", "soporte paragolpes", "paragolpes delantero"],
        "soporte paragolpes"
    ),
    (
        "guard del izq",
        ["guardabarro delantero izquierdo", "guardabarro trasero izquierdo",
            "guardabarro delantero derecho"],
        "guardabarro delantero izquierdo"
    ),
]


def test_normalization():
    """Test the normalization functions."""
    print("\n=== Testing Text Normalization ===")
    print(f"{'Original Text':<30} | {'Standard Normalization':<30} | {'Fuzzy Normalization':<30} | {'Match?':<6}")
    print("-" * 100)

    for original, expected_standard, expected_fuzzy in TEST_CASES:
        standard_result = normalize_text(original)
        fuzzy_result = normalize_text(original, use_fuzzy=True)

        standard_match = "✓" if standard_result == expected_standard else "✗"
        fuzzy_match = "✓" if fuzzy_result == expected_fuzzy else "✗"
        match = "✓" if standard_match == "✓" and fuzzy_match == "✓" else "✗"

        print(
            f"{original:<30} | {standard_result:<30} | {fuzzy_result:<30} | {match:<6}")

        if standard_match == "✗":
            print(
                f"  Standard normalization failed: Expected '{expected_standard}', got '{standard_result}'")
        if fuzzy_match == "✗":
            print(
                f"  Fuzzy normalization failed: Expected '{expected_fuzzy}', got '{fuzzy_result}'")


def test_fuzzy_matching():
    """Test the fuzzy matching functions."""
    print("\n=== Testing Fuzzy Matching ===")
    print(f"{'Query':<20} | {'Best Match':<30} | {'Similarity':<10} | {'Expected':<30} | {'Match?':<6}")
    print("-" * 100)

    for query, candidates, expected in FUZZY_MATCH_TESTS:
        # Normalize the query and candidates
        normalized_query = normalize_text(query, use_fuzzy=True)
        normalized_candidates = [normalize_text(
            c, use_fuzzy=True) for c in candidates]

        # Find the best match
        match_result = find_best_match(normalized_query, normalized_candidates)

        if match_result:
            best_match, similarity = match_result
            expected_idx = candidates.index(expected)
            expected_normalized = normalized_candidates[expected_idx]
            match = "✓" if best_match == expected_normalized else "✗"

            print(
                f"{query:<20} | {best_match:<30} | {similarity:<10.2f} | {expected_normalized:<30} | {match:<6}")

            if match == "✗":
                print(
                    f"  Fuzzy matching failed: Expected '{expected_normalized}', got '{best_match}'")
        else:
            print(
                f"{query:<20} | {'No match found':<30} | {'-':<10} | {expected:<30} | {'✗':<6}")


def test_expand_abbreviations():
    """Test the expand_abbreviations function."""
    print("\n=== Testing Abbreviation Expansion ===")
    print(f"{'Original Text':<30} | {'Expanded Text':<30}")
    print("-" * 62)

    test_cases = [
        "izq",
        "del",
        "der",
        "tra",
        "parag",
        "guard",
        "sop parag del izq",
        "faro tra der",
    ]

    for text in test_cases:
        expanded = expand_abbreviations(text)
        print(f"{text:<30} | {expanded:<30}")


def test_split_compound_words():
    """Test the split_compound_words function."""
    print("\n=== Testing Compound Word Splitting ===")
    print(f"{'Original Text':<30} | {'Split Text':<30}")
    print("-" * 62)

    test_cases = [
        "guardabarroizq",
        "paragolpesdelantero",
        "farolaizquierda",
        "puertadelantera",
        "soporteparagolpes",
        "guardabarrodelanteroizquierdo",
    ]

    for text in test_cases:
        split = split_compound_words(text)
        print(f"{text:<30} | {split:<30}")


if __name__ == "__main__":
    print("=== Fuzzy Matching Test Script ===")
    test_normalization()
    test_fuzzy_matching()
    test_expand_abbreviations()
    test_split_compound_words()
    print("\nAll tests completed.")
