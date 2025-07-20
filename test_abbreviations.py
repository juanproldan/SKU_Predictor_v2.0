#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

try:
    from utils.text_utils import normalize_text
except ImportError:
    # Fallback normalize function if text_utils not available
    def normalize_text(text, **kwargs):
        return text.lower().strip()

def create_abbreviated_version(description: str) -> str:
    """
    Create abbreviated version of description to match database format.
    Database uses heavily abbreviated forms like 'paragolpes del' instead of 'paragolpes delantero'.
    Uses centralized text normalization for consistency.
    """
    # Use centralized normalization (case-insensitive, handles synonyms, etc.)
    desc = normalize_text(description, expand_linguistic_variations=True)
    
    # Common abbreviations found in database
    abbreviations = {
        'delantero': 'del',
        'delantera': 'del', 
        'trasero': 'tra',
        'trasera': 'tra',
        'izquierdo': 'i',
        'izquierda': 'i',
        'derecho': 'd',
        'derecha': 'd',
        'superior': 'sup',
        'inferior': 'inf',
        'anterior': 'ant',
        'posterior': 'post',
        'paragolpes': 'paragolpes',  # Keep as is
        'guardafango': 'guardafango',  # Keep as is
        'absorbedor de impactos': 'absorbimpacto',
        'electroventilador': 'electrovent',
        'antiniebla': 'antiniebla',
        'frontal': 'frontal'
    }
    
    # Apply abbreviations
    for full_form, abbrev in abbreviations.items():
        desc = desc.replace(full_form, abbrev)
    
    # Remove common words that might not be in database
    remove_words = ['de', 'la', 'el', 'los', 'las']
    words = desc.split()
    words = [w for w in words if w not in remove_words]
    
    return ' '.join(words)

# Test our abbreviation function
test_descriptions = [
    'absorbedor de impactos paragolpes delantero',
    'electroventilador radiador',
    'guardafango izquierdo',
    'luz antiniebla delantera derecha',
    'luz antiniebla delantera izquierda',
    'rejilla frontal'
]

print("=== TESTING ABBREVIATION FUNCTION ===")
for desc in test_descriptions:
    abbreviated = create_abbreviated_version(desc)
    print(f"Original: '{desc}'")
    print(f"Abbreviated: '{abbreviated}'")
    print()

# Compare with what we found in database
print("=== DATABASE ENTRIES WE FOUND ===")
database_entries = [
    'paragolpes del',
    'farola d',
    'farola i', 
    'guardafango deld',
    'electrovent radiador'  # hypothetical
]

for entry in database_entries:
    print(f"Database has: '{entry}'")

print("\n=== COMPARISON ===")
print("Our abbreviations should match database patterns!")
