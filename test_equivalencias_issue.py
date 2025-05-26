#!/usr/bin/env python3
"""
Test script to reproduce and analyze the Equivalencias synonym issue.

Problem:
- "FAROLA IZQUIERDA" returns high confidence (~0.6)
- "FAROLA IZQ" returns low confidence (~0.08)
- Synonyms like "IZQUIERDA" = "IZQ" = "IZ" are not being properly applied

This script will:
1. Load the current Equivalencias.xlsx file
2. Analyze the synonym mappings
3. Test the normalization process for the problematic inputs
4. Identify where the synonym expansion is failing
"""

import pandas as pd
import os
import sys
import re
import unicodedata

def normalize_text(text: str, use_fuzzy: bool = False) -> str:
    """Simple text normalization function."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove leading/trailing whitespace
    text = text.strip()

    # Normalize accented characters
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Remove common punctuation (keeps alphanumeric characters and spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Standardize internal whitespace (multiple spaces to one)
    text = re.sub(r'\s+', ' ', text)

    return text

def load_equivalencias_data(file_path: str) -> dict:
    """Load equivalencias mapping from Excel file (same as main app)."""
    print(f"Loading equivalencias from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: Equivalencias file not found at {file_path}")
        return {}

    try:
        df = pd.read_excel(file_path, sheet_name=0)
        equivalencias_map = {}

        print(f"Equivalencias file structure:")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        print(f"  Sample data:")
        print(df.head())
        print()

        for index, row in df.iterrows():
            equivalencia_row_id = index + 1
            print(f"Row {equivalencia_row_id}:")
            row_terms = []
            for col_name in df.columns:
                term = row[col_name]
                if pd.notna(term) and str(term).strip():
                    normalized_term = normalize_text(str(term))
                    if normalized_term:
                        equivalencias_map[normalized_term] = equivalencia_row_id
                        row_terms.append(f"'{term}' -> '{normalized_term}'")
            print(f"  Terms: {', '.join(row_terms)}")
            print()

        print(f"Total normalized term mappings: {len(equivalencias_map)}")
        return equivalencias_map

    except Exception as e:
        print(f"Error loading Equivalencias.xlsx: {e}")
        return {}

def test_synonym_processing(equivalencias_map: dict):
    """Test the synonym processing for the problematic inputs."""
    test_inputs = [
        "FAROLA IZQUIERDA",
        "FAROLA IZQ",
        "FAROLA IZ",
        "IZQUIERDA",
        "IZQ",
        "IZ"
    ]

    print("=== SYNONYM PROCESSING TEST ===")
    for test_input in test_inputs:
        print(f"\nTesting: '{test_input}'")

        # Standard normalization
        standard_normalized = normalize_text(test_input)
        eq_id_standard = equivalencias_map.get(standard_normalized)
        print(f"  Standard normalized: '{standard_normalized}' -> EqID: {eq_id_standard}")

        # Fuzzy normalization
        fuzzy_normalized = normalize_text(test_input, use_fuzzy=True)
        eq_id_fuzzy = equivalencias_map.get(fuzzy_normalized)
        print(f"  Fuzzy normalized: '{fuzzy_normalized}' -> EqID: {eq_id_fuzzy}")

        # Check if any part of the input matches
        words = test_input.split()
        print(f"  Word analysis:")
        for word in words:
            word_normalized = normalize_text(word)
            word_eq_id = equivalencias_map.get(word_normalized)
            print(f"    '{word}' -> '{word_normalized}' -> EqID: {word_eq_id}")

def analyze_synonym_groups(equivalencias_map: dict):
    """Analyze the synonym groups to understand the structure."""
    print("\n=== SYNONYM GROUPS ANALYSIS ===")

    # Group by equivalencia_row_id
    groups = {}
    for term, eq_id in equivalencias_map.items():
        if eq_id not in groups:
            groups[eq_id] = []
        groups[eq_id].append(term)

    print(f"Total synonym groups: {len(groups)}")

    # Look for groups containing our test terms
    target_terms = ['izquierda', 'izq', 'iz', 'farola']

    for eq_id, terms in groups.items():
        # Check if any target terms are in this group
        has_target = any(any(target in term for target in target_terms) for term in terms)
        if has_target:
            print(f"\nGroup {eq_id}: {terms}")

def main():
    print("=== EQUIVALENCIAS SYNONYM ISSUE ANALYSIS ===\n")

    # Change to project directory
    os.chdir(r'c:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0')
    print(f"Working directory: {os.getcwd()}")

    # Load equivalencias data
    equivalencias_file = 'Source_Files/Equivalencias.xlsx'
    equivalencias_map = load_equivalencias_data(equivalencias_file)

    if not equivalencias_map:
        print("No equivalencias data loaded. Exiting.")
        return

    # Test synonym processing
    test_synonym_processing(equivalencias_map)

    # Analyze synonym groups
    analyze_synonym_groups(equivalencias_map)

    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
