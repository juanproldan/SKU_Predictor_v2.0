#!/usr/bin/env python3
"""
Test script to verify the synonym expansion functionality.
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

def load_equivalencias_and_create_synonym_map(file_path: str):
    """Load equivalencias and create synonym expansion map."""
    print(f"Loading equivalencias from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return {}, {}
    
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        equivalencias_map = {}
        synonym_expansion_map = {}
        
        print(f"Equivalencias file has {len(df)} rows and {len(df.columns)} columns")
        
        for index, row in df.iterrows():
            equivalencia_row_id = index + 1
            row_terms = []
            
            # First pass: collect all normalized terms in this row
            for col_name in df.columns:
                term = row[col_name]
                if pd.notna(term) and str(term).strip():
                    normalized_term = normalize_text(str(term))
                    if normalized_term:
                        row_terms.append(normalized_term)
                        equivalencias_map[normalized_term] = equivalencia_row_id
            
            # Second pass: create synonym mappings (all terms map to the first/canonical term)
            if row_terms:
                canonical_term = row_terms[0]  # Use first term as canonical
                print(f"Row {equivalencia_row_id}: {row_terms} -> canonical: '{canonical_term}'")
                for term in row_terms:
                    synonym_expansion_map[term] = canonical_term
        
        print(f"Created {len(equivalencias_map)} equivalencia mappings")
        print(f"Created {len(synonym_expansion_map)} synonym expansion mappings")
        
        return equivalencias_map, synonym_expansion_map
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}, {}

def expand_synonyms(text: str, synonym_expansion_map: dict) -> str:
    """Expand synonyms in text."""
    if not text or not synonym_expansion_map:
        return text
    
    # Split text into words
    words = text.split()
    expanded_words = []
    
    for word in words:
        # Normalize the word first
        normalized_word = normalize_text(word)
        
        # Check if this word has a canonical form
        if normalized_word in synonym_expansion_map:
            canonical_form = synonym_expansion_map[normalized_word]
            expanded_words.append(canonical_form)
            print(f"    Synonym expansion: '{word}' -> '{canonical_form}'")
        else:
            expanded_words.append(normalized_word)
    
    expanded_text = ' '.join(expanded_words)
    return expanded_text

def test_synonym_expansion():
    """Test the synonym expansion with problematic inputs."""
    print("=== SYNONYM EXPANSION TEST ===\n")
    
    # Load equivalencias data
    equivalencias_file = 'Source_Files/Equivalencias.xlsx'
    equivalencias_map, synonym_expansion_map = load_equivalencias_and_create_synonym_map(equivalencias_file)
    
    if not synonym_expansion_map:
        print("No synonym data loaded. Exiting.")
        return
    
    # Test inputs
    test_inputs = [
        "FAROLA IZQUIERDA",
        "FAROLA IZQ", 
        "FAROLA IZ",
        "IZQUIERDA",
        "IZQ",
        "IZ"
    ]
    
    print(f"\n=== TESTING SYNONYM EXPANSION ===")
    for test_input in test_inputs:
        print(f"\nTesting: '{test_input}'")
        
        # Apply synonym expansion
        expanded = expand_synonyms(test_input, synonym_expansion_map)
        print(f"  Expanded: '{expanded}'")
        
        # Normalize the expanded form
        normalized = normalize_text(expanded)
        print(f"  Normalized: '{normalized}'")
        
        # Check equivalencia ID
        eq_id = equivalencias_map.get(normalized)
        print(f"  Equivalencia ID: {eq_id}")
        
        print(f"  RESULT: '{test_input}' -> '{expanded}' -> '{normalized}' (EqID: {eq_id})")

if __name__ == "__main__":
    test_synonym_expansion()
