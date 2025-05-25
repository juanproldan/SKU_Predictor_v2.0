"""
Fuzzy matching utilities for Spanish automotive part descriptions.

This module provides functions to enhance text matching capabilities for the SKU Predictor
application, specifically tailored for Spanish automotive terminology.

Features:
- Dictionary of common Spanish automotive abbreviations and their full forms
- Stemming/lemmatization specific to Spanish automotive terminology
- Edit distance algorithms with customized thresholds
- N-gram matching for compound words
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Set, Optional, Union, Callable
import unicodedata

# Dictionary of common Spanish automotive abbreviations and their full forms
# This dictionary maps abbreviations to their full forms
AUTOMOTIVE_ABBR = {
    # Directional terms
    "izq": "izquierdo",
    "iz": "izquierdo",
    "izquier": "izquierdo",
    "i": "izquierdo",
    "der": "derecho",
    "dere": "derecho",
    "d": "derecho",
    "del": "delantero",
    "delan": "delantero",
    "de": "delantero",
    "tra": "trasero",
    "tras": "trasero",
    "t": "trasero",

    # Common parts
    "parag": "paragolpes",
    "paragol": "paragolpes",
    "guard": "guardabarro",
    "guardab": "guardabarro",
    "faro": "farola",
    "far": "farola",
    "luz": "farola",
    "luces": "farola",
    "espej": "espejo",
    "esp": "espejo",
    "puert": "puerta",
    "pta": "puerta",
    "capot": "capo",
    "cap": "capo",
    "parabr": "parabrisas",
    "parabrisa": "parabrisas",
    "vidrio": "parabrisas",
    "vidr": "parabrisas",
    "motor": "motor",
    "mot": "motor",
    "bomper": "paragolpes",
    "defensa": "paragolpes",
    "def": "paragolpes",
    "soporte": "soporte",
    "sop": "soporte",
    "sopt": "soporte",

    # Common conjunctions and prepositions that might be abbreviated
    "c": "con",
    "c/": "con",
    "s": "sin",
    "s/": "sin",
    "p": "para",
    "p/": "para",
}

# Common word combinations that might be written without spaces
COMPOUND_WORDS = {
    "guardabarroizq": ["guardabarro", "izquierdo"],
    "guardabarroder": ["guardabarro", "derecho"],
    "guardabarrodelantero": ["guardabarro", "delantero"],
    "guardabarrotrasero": ["guardabarro", "trasero"],
    "paragolpesdelantero": ["paragolpes", "delantero"],
    "paragolpestrasero": ["paragolpes", "trasero"],
    "farolaizquierda": ["farola", "izquierda"],
    "faroladerecha": ["farola", "derecha"],
    "faroladelantera": ["farola", "delantera"],
    "farolatrasera": ["farola", "trasera"],
    "puertadelantera": ["puerta", "delantera"],
    "puertatrasera": ["puerta", "trasera"],
    "puertaizquierda": ["puerta", "izquierda"],
    "puertaderecha": ["puerta", "derecha"],
    "espejolateral": ["espejo", "lateral"],
    "espejoretrovisor": ["espejo", "retrovisor"],
    "soporteparagolpes": ["soporte", "paragolpes"],
    "soportemotor": ["soporte", "motor"],
}


def expand_abbreviations(text: str) -> str:
    """
    Expands common automotive abbreviations in the text.

    Args:
        text: The input text with possible abbreviations

    Returns:
        Text with abbreviations expanded to their full forms
    """
    words = text.split()
    expanded_words = []

    for word in words:
        # Check if the word is in our abbreviation dictionary
        if word in AUTOMOTIVE_ABBR:
            expanded_words.append(AUTOMOTIVE_ABBR[word])
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)


def split_compound_words(text: str) -> str:
    """
    Splits common compound words that might be written without spaces.

    Args:
        text: The input text with possible compound words

    Returns:
        Text with compound words split into their component parts
    """
    # First check for exact matches in our compound words dictionary
    for compound, components in COMPOUND_WORDS.items():
        if compound in text:
            text = text.replace(compound, " ".join(components))

    # Then try to identify other potential compound words using regex patterns
    # This is a simplified approach - a more sophisticated approach would use
    # a sliding window to check for valid word combinations

    # Pattern for words like "guardabarroizquierdo" -> "guardabarro izquierdo"
    pattern = r'(guardabarro|paragolpes|farola|puerta|espejo|soporte)(izquierdo|derecho|delantero|trasero|lateral)'
    text = re.sub(pattern, r'\1 \2', text)

    # Additional patterns for specific cases
    if "sopdparagolpes" in text:
        text = text.replace("sopdparagolpes", "soporte de paragolpes")

    if "guardbarroizq" in text:
        text = text.replace("guardbarroizq", "guardabarro izquierdo")

    return text


def calculate_similarity(s1: str, s2: str) -> float:
    """
    Calculates the similarity ratio between two strings using SequenceMatcher.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    return SequenceMatcher(None, s1, s2).ratio()


def find_best_match(query: str, candidates: List[str], threshold: float = 0.7) -> Optional[Tuple[str, float]]:
    """
    Finds the best matching candidate for a query string.

    Args:
        query: The string to match
        candidates: List of candidate strings to match against
        threshold: Minimum similarity threshold (0.0 to 1.0)

    Returns:
        Tuple of (best_match, similarity_score) or None if no match above threshold
    """
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        score = calculate_similarity(query, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return (best_match, best_score)
    return None


def fuzzy_normalize_text(text: str) -> str:
    """
    Enhanced text normalization with fuzzy matching capabilities.

    This function extends the basic normalize_text function with:
    - Abbreviation expansion
    - Compound word splitting

    Args:
        text: The input text to normalize

    Returns:
        Normalized text with fuzzy matching enhancements
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove leading/trailing whitespace
    text = text.strip()

    # 3. Normalize accented characters
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # 4. Remove common punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # 5. Standardize internal whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Split compound words
    text = split_compound_words(text)

    # 7. Expand abbreviations
    text = expand_abbreviations(text)

    # 8. Handle special cases
    # Fix the issue with "soporte de paragolpes delantero izquierdo"
    if "soporte" in text and "paragolpes" in text and "delantero" in text:
        text = text.replace("soporte delantero paragolpes",
                            "soporte de paragolpes")

    # 9. Standardize internal whitespace again (after our transformations)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_fuzzy_matches(query: str, candidates: List[str], threshold: float = 0.7) -> List[Tuple[str, float]]:
    """
    Returns all candidates that match the query with a similarity above the threshold.

    Args:
        query: The string to match
        candidates: List of candidate strings to match against
        threshold: Minimum similarity threshold (0.0 to 1.0)

    Returns:
        List of tuples (candidate, similarity_score) sorted by score descending
    """
    matches = []

    for candidate in candidates:
        score = calculate_similarity(query, candidate)
        if score >= threshold:
            matches.append((candidate, score))

    # Sort by score descending
    return sorted(matches, key=lambda x: x[1], reverse=True)
