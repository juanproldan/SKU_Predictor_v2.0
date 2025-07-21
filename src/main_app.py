import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  # For showing error popups
import os
import pandas as pd
# import requests # No longer needed
import sqlite3  # For database connection
from collections import defaultdict  # For counting frequencies
import datetime  # For timestamping Maestro entries
import json  # For loading consolidado
import joblib  # To load trained models
import numpy as np  # For model input reshaping
import re  # For VIN validation
import torch  # For PyTorch
# Import our PyTorch model implementation
try:
    # Try relative imports first (for PyInstaller)
    from models.sku_nn_pytorch import load_model, predict_sku
    from utils.text_utils import normalize_text
    from utils.dummy_tokenizer import DummyTokenizer
    from train_vin_predictor import extract_vin_features, decode_year
except ImportError:
    try:
        # Fallback for package execution
        from .models.sku_nn_pytorch import load_model, predict_sku
        from .utils.text_utils import normalize_text
        from .utils.dummy_tokenizer import DummyTokenizer
        from .train_vin_predictor import extract_vin_features, decode_year
    except ImportError:
        # Final fallback - add src to path and try again
        import sys
        import os
        src_path = os.path.dirname(os.path.abspath(__file__))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from models.sku_nn_pytorch import load_model, predict_sku
        from utils.text_utils import normalize_text
        from utils.dummy_tokenizer import DummyTokenizer
        from train_vin_predictor import extract_vin_features, decode_year

import sys

# --- Determine Project Root Path ---
# Get the directory of the current script (src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume the project root is one level up from 'src'
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# --- Configuration (using resource path) ---
DEFAULT_EQUIVALENCIAS_PATH = get_resource_path(os.path.join(
    "Source_Files", "Equivalencias.xlsx"))
DEFAULT_MAESTRO_PATH = get_resource_path(os.path.join("data", "Maestro.xlsx"))
DEFAULT_DB_PATH = get_resource_path(os.path.join("data", "fixacar_history.db"))
MODEL_DIR = get_resource_path("models")
SKU_NN_MODEL_DIR = os.path.join(MODEL_DIR, "sku_nn")

# Define pattern and count for loading VIN details from chunks (Used by load_vin_details_from_chunks)
# This path might also need adjustment if it's not relative to CWD
# Keeping this relative for now, assuming it's handled elsewhere or CWD is intended for these specific chunks
CONSOLIDADO_CHUNK_PATTERN_FOR_VIN_LOAD = "Consolidado_chunk_{}.json"
NUM_CONSOLIDADO_CHUNKS_FOR_VIN_LOAD = 10


# In-memory data stores
equivalencias_map_global = {}
synonym_expansion_map_global = {}  # New: maps synonyms to equivalence group IDs
maestro_data_global = []  # This will hold the list of dictionaries
# VIN details lookup is replaced by models

# Loaded Models and Encoders
model_maker = None
encoder_x_maker = None
encoder_y_maker = None
model_year = None
encoder_x_year = None
encoder_y_year = None
model_series = None
encoder_x_series = None
encoder_y_series = None

# SKU NN Model and Encoders/Tokenizer
sku_nn_model = None
sku_nn_encoder_make = None
sku_nn_encoder_model_year = None
sku_nn_encoder_series = None
sku_nn_tokenizer_desc = None  # Assuming description is an input
sku_nn_encoder_sku = None
SKU_NN_MODEL_DIR = os.path.join(MODEL_DIR, "sku_nn")


class FixacarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fixacar SKU Finder v1.0 (with VIN Predictor)")
        self.root.geometry("800x750")  # Increased height

        # Maximize the window on startup to ensure all buttons are visible
        self.root.state('zoomed')  # Windows equivalent of maximized

        # Load initial data and models
        self.load_all_data_and_models()

        # Setup UI
        self.create_widgets()

    def load_all_data_and_models(self):
        """Loads data files and trained prediction models on startup."""
        global equivalencias_map_global, synonym_expansion_map_global, maestro_data_global
        global model_maker, encoder_x_maker, encoder_y_maker
        global model_year, encoder_x_year, encoder_y_year
        global model_series, encoder_x_series, encoder_y_series

        print("--- Loading Application Data & Models ---")
        equivalencias_map_global = self.load_equivalencias_data(
            DEFAULT_EQUIVALENCIAS_PATH)
        maestro_data_global = self.load_maestro_data(
            DEFAULT_MAESTRO_PATH, equivalencias_map_global)

        # Load Models
        print("Loading VIN prediction models...")
        try:
            model_maker = joblib.load(os.path.join(
                MODEL_DIR, 'vin_maker_model.joblib'))
            encoder_x_maker = joblib.load(os.path.join(
                MODEL_DIR, 'vin_maker_encoder_x.joblib'))
            encoder_y_maker = joblib.load(os.path.join(
                MODEL_DIR, 'vin_maker_encoder_y.joblib'))
            print("  Maker model loaded.")

            model_year = joblib.load(os.path.join(
                MODEL_DIR, 'vin_year_model.joblib'))
            encoder_x_year = joblib.load(os.path.join(
                MODEL_DIR, 'vin_year_encoder_x.joblib'))
            encoder_y_year = joblib.load(os.path.join(
                MODEL_DIR, 'vin_year_encoder_y.joblib'))
            print("  Year model loaded.")

            model_series = joblib.load(os.path.join(
                MODEL_DIR, 'vin_series_model.joblib'))
            encoder_x_series = joblib.load(os.path.join(
                MODEL_DIR, 'vin_series_encoder_x.joblib'))
            encoder_y_series = joblib.load(os.path.join(
                MODEL_DIR, 'vin_series_encoder_y.joblib'))
            print("  Series model loaded.")
            print("All models loaded successfully.")

        except FileNotFoundError as e:
            print(
                f"Error loading model file: {e}. VIN prediction will not work.")
            messagebox.showerror(
                "Model Loading Error", f"Could not load model file: {e}\nPlease ensure models are trained and present in the '{MODEL_DIR}' directory.")
            # Set models to None so prediction attempts fail gracefully
            model_maker, model_year, model_series = None, None, None
        except Exception as e:
            print(f"An unexpected error occurred loading models: {e}")
            messagebox.showerror(
                "Model Loading Error", f"An unexpected error occurred loading models: {e}")
            model_maker, model_year, model_series = None, None, None

        # Load SKU NN Model and preprocessors
        global sku_nn_model, sku_nn_encoder_make, sku_nn_encoder_model_year, sku_nn_encoder_series, sku_nn_tokenizer_desc, sku_nn_encoder_sku
        print("Loading SKU NN model and preprocessors...")
        try:
            # Only load the optimized PyTorch model and encoders
            sku_nn_model_path = os.path.join(
                SKU_NN_MODEL_DIR, 'sku_nn_model_pytorch_optimized.pth')

            sku_nn_encoder_make = joblib.load(os.path.join(
                SKU_NN_MODEL_DIR, 'encoder_Make.joblib'))
            print("  SKU NN Make encoder loaded.")
            sku_nn_encoder_model_year = joblib.load(os.path.join(
                SKU_NN_MODEL_DIR, 'encoder_Model Year.joblib'))
            print("  SKU NN Model Year encoder loaded.")
            sku_nn_encoder_series = joblib.load(os.path.join(
                SKU_NN_MODEL_DIR, 'encoder_Series.joblib'))
            print("  SKU NN Series encoder loaded.")
            try:
                sku_nn_tokenizer_desc = joblib.load(os.path.join(
                    SKU_NN_MODEL_DIR, 'tokenizer.joblib'))
                print("  SKU NN Description tokenizer loaded.")
            except Exception as e:
                print(f"  Error loading tokenizer: {e}")
                try:
                    from utils.pytorch_tokenizer import PyTorchTokenizer
                    sku_nn_tokenizer_desc = PyTorchTokenizer(
                        num_words=10000, oov_token="<OOV>")
                    print("  Using PyTorchTokenizer instead.")
                except ImportError:
                    from utils.dummy_tokenizer import DummyTokenizer
                    sku_nn_tokenizer_desc = DummyTokenizer(
                        num_words=10000, oov_token="<OOV>")
                    print("  Using DummyTokenizer instead.")

            sku_nn_encoder_sku = joblib.load(os.path.join(
                SKU_NN_MODEL_DIR, 'encoder_sku.joblib'))
            print("  SKU NN SKU encoder loaded.")

            # Now try to load the optimized PyTorch model
            if os.path.exists(sku_nn_model_path):
                sku_nn_model, _ = load_model(SKU_NN_MODEL_DIR)
                if sku_nn_model:
                    print("  SKU NN Optimized PyTorch model loaded successfully.")
                else:
                    print("  Failed to load SKU NN Optimized PyTorch model.")
            else:
                print(
                    f"  SKU NN Optimized PyTorch model file not found at {sku_nn_model_path}")
                print(
                    "  Note: You need to train and save an optimized PyTorch model first.")
                sku_nn_model = None

        except FileNotFoundError as e:
            print(
                f"Error loading SKU NN model file: {e}. SKU NN prediction will not work.")
            messagebox.showerror("SKU NN Model Error",
                                 f"Could not load SKU NN model file: {e}")
            sku_nn_model = None  # Ensure it's None if loading fails
        except Exception as e:
            print(f"An unexpected error occurred loading SKU NN models: {e}")
            messagebox.showerror(
                "SKU NN Model Error", f"An unexpected error occurred loading SKU NN models: {e}")
            sku_nn_model = None

        # DB connection will be established when needed for search
        print("--- Application Data & Model Loading Complete ---")

    def expand_synonyms(self, text: str) -> str:
        """
        Global synonym expansion function that preprocesses text by replacing
        industry-specific synonyms with their equivalence group representatives before any prediction method.

        This function ONLY handles Equivalencias.xlsx synonyms (industry-specific terms).
        Linguistic variations (abbreviations, gender, plurals) are handled automatically
        by the normalize_text function.

        This ensures ALL prediction sources (Maestro, Database, Neural Network)
        receive the same normalized input after synonym expansion.
        """
        if not text or not synonym_expansion_map_global:
            return text

        # First normalize text (this handles abbreviations, gender, plurals automatically)
        normalized_text = normalize_text(text, expand_linguistic_variations=True)
        words = normalized_text.split()
        expanded_words = []

        for word in words:
            # Check if this word has an industry-specific synonym in Equivalencias (CASE-INSENSITIVE)
            word_lower = word.lower()
            if word_lower in synonym_expansion_map_global:
                group_id = synonym_expansion_map_global[word_lower]
                # Use the group_id as a consistent representation
                group_representative = f"GROUP_{group_id}"
                expanded_words.append(group_representative)
                print(f"    Industry synonym: '{word}' -> '{group_representative}' (Group ID: {group_id})")
            else:
                expanded_words.append(word)

        expanded_text = ' '.join(expanded_words)
        return expanded_text

    def calculate_frequency_based_confidence(self, frequency: int, prediction_type: str = "DB") -> float:
        """
        Calculate confidence based on absolute frequency of SKU occurrences in database.

        Updated confidence ranges for new priority system:
        - 1-2 occurrences: 0.4-0.45 (40-45%) - very low confidence
        - 3-9 occurrences: 0.45-0.55 (45-55%) - low confidence
        - 10-19 occurrences: 0.55-0.7 (55-70%) - medium confidence
        - 20+ occurrences: 0.7-0.8 (70-80%) - high confidence

        Args:
            frequency: Number of times this SKU appears in database for this combination
            prediction_type: Type of prediction for confidence adjustment

        Returns:
            Confidence score between 0.4 and 0.8 (40-80%)
        """
        if frequency < 3:
            # Very low confidence for rare occurrences (likely errors)
            base_confidence = 0.4 + 0.025 * frequency  # 0.4-0.45 range
        elif frequency < 10:
            # Low confidence for insufficient data
            base_confidence = 0.45 + 0.01 * frequency  # 0.45-0.54 range
        elif frequency < 20:
            # Medium confidence for moderate data
            base_confidence = 0.55 + 0.0075 * frequency  # 0.55-0.7 range
        else:
            # High confidence for reliable data (20+ occurrences)
            # Cap at 50+ occurrences to avoid overconfidence
            capped_frequency = min(frequency, 50)
            base_confidence = 0.7 + 0.002 * capped_frequency  # 0.7-0.8 range

        # Slight adjustment based on prediction type
        if prediction_type == "DB (4-param Exact)":
            multiplier = 1.0  # Full confidence for exact matches
        elif prediction_type.startswith("DB (Unified Fuzzy"):
            multiplier = 0.9  # Slightly lower for fuzzy matches
        elif prediction_type.startswith("DB (3-param"):
            multiplier = 0.8  # Lower for 3-parameter matches
        else:
            multiplier = 0.7  # Lowest for fallback matches

        final_confidence = round(base_confidence * multiplier, 3)

        print(f"    Frequency-based confidence: {frequency} occurrences → {final_confidence} confidence")
        return final_confidence

    def apply_consensus_logic(self, sku_frequency_pairs: list, min_consensus_ratio: float = 0.6) -> list:
        """
        Apply consensus logic to filter out minority/outlier SKUs.

        If we have: 25 × "SKU123", 3 × "SKU456", 1 × "SKU789"
        Only return SKUs that represent significant consensus, not obvious errors.

        Args:
            sku_frequency_pairs: List of (sku, frequency) tuples
            min_consensus_ratio: Minimum ratio of total occurrences for a SKU to be considered

        Returns:
            Filtered list of (sku, frequency) tuples with only consensus SKUs
        """
        if not sku_frequency_pairs:
            return []

        total_occurrences = sum(freq for _, freq in sku_frequency_pairs)

        # Sort by frequency (highest first)
        sorted_pairs = sorted(sku_frequency_pairs, key=lambda x: x[1], reverse=True)

        consensus_skus = []
        for sku, frequency in sorted_pairs:
            ratio = frequency / total_occurrences

            print(f"    Consensus analysis: {sku} appears {frequency}/{total_occurrences} times ({ratio:.2%})")

            # Include SKUs that meet minimum consensus threshold
            if ratio >= min_consensus_ratio:
                consensus_skus.append((sku, frequency))
                print(f"      ✅ Included: Strong consensus ({ratio:.2%} ≥ {min_consensus_ratio:.1%})")
            elif frequency >= 20:  # Always include high-frequency SKUs even if ratio is low
                consensus_skus.append((sku, frequency))
                print(f"      ✅ Included: High frequency ({frequency} ≥ 20 occurrences)")
            else:
                print(f"      ❌ Excluded: Weak consensus ({ratio:.2%} < {min_consensus_ratio:.1%}, freq: {frequency})")

        print(f"    Consensus result: {len(consensus_skus)}/{len(sorted_pairs)} SKUs passed consensus filter")
        return consensus_skus

    def unified_text_preprocessing(self, text: str) -> str:
        """
        Unified Text Preprocessing Pipeline for ALL text comparisons in the SKU prediction system.

        This ensures that BOTH input descriptions AND target comparison texts (from Database/Maestro)
        receive identical preprocessing, eliminating false penalties for linguistically equivalent terms.

        Pipeline:
        1. Synonym Expansion: Apply Equivalencias.xlsx industry synonyms
        2. Linguistic Normalization: Expand abbreviations, handle gender agreement, plurals/singulars
        3. Text Normalization: Convert to lowercase, remove extra spaces, standardize punctuation

        Example:
        - Input: "FAROLA IZQ" → "GROUP_1001"
        - Target: "FAROLA IZQUIERDA" → "GROUP_1001"
        - Result: Perfect match (1.0 similarity) instead of penalized fuzzy match (0.85)
        """
        if not text or not text.strip():
            return ""

        # Step 1: Apply synonym expansion (industry-specific terms from Equivalencias.xlsx)
        expanded_text = self.expand_synonyms(text)

        # Step 2: Apply comprehensive linguistic normalization
        # This handles abbreviations, gender agreement, plurals/singulars
        normalized_text = normalize_text(expanded_text, expand_linguistic_variations=True)

        # Step 3: Final text normalization (lowercase, spaces, punctuation)
        final_text = normalized_text.lower().strip()

        print(f"    Unified preprocessing: '{text}' → '{final_text}'")
        return final_text

    def create_abbreviated_version(self, description: str) -> str:
        """
        Create abbreviated version of description to match database format.
        Database uses heavily abbreviated forms like 'paragolpes del' instead of 'paragolpes delantero'.

        Based on database analysis, common patterns are:
        - 'paragolpes del' (25 records)
        - 'farola d' (18 records)
        - 'farola i' (15 records)
        - 'guardafango deld' (13 records)
        - 'absorbimpacto del' (11 records)
        """
        desc = description.lower()

        # Apply specific transformations based on database patterns
        # Handle compound terms first
        if 'absorbedor de impactos' in desc:
            desc = desc.replace('absorbedor de impactos', 'absorbimpacto')

        if 'electroventilador' in desc:
            desc = desc.replace('electroventilador', 'electrovent')

        # Handle luz -> farola transformation for lights
        if 'luz antiniebla' in desc:
            desc = desc.replace('luz antiniebla', 'farola')
        elif 'luz' in desc:
            desc = desc.replace('luz', 'farola')

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
            'posterior': 'post'
        }

        # Apply abbreviations
        for full_form, abbrev in abbreviations.items():
            desc = desc.replace(full_form, abbrev)

        # Remove common words that might not be in database (CASE-INSENSITIVE)
        remove_words = ['de', 'la', 'el', 'los', 'las', 'antiniebla']
        words = desc.split()
        words = [w for w in words if w.lower() not in remove_words]

        return ' '.join(words)

    def _calculate_description_similarity(self, query_desc: str, db_desc: str) -> float:
        """
        Calculate similarity between query description and database description.
        Handles gender variations, abbreviations, and partial matches.

        Examples:
        - 'farola derecho' vs 'farola derecha' → high similarity (gender variation)
        - 'farola der' vs 'farola derecha' → high similarity (abbreviation)
        - 'farola dere' vs 'farola derecha' → high similarity (partial)

        CASE-INSENSITIVE: All comparisons are done in lowercase.
        """
        if not query_desc or not db_desc:
            return 0.0

        # Convert to lowercase for case-insensitive comparison
        query_desc = query_desc.lower()
        db_desc = db_desc.lower()

        # Exact match
        if query_desc == db_desc:
            return 1.0

        # Split into words for detailed comparison
        query_words = query_desc.split()
        db_words = db_desc.split()

        # If different number of words, use sequence matching
        if len(query_words) != len(db_words):
            from difflib import SequenceMatcher
            return SequenceMatcher(None, query_desc, db_desc).ratio()

        # Word-by-word comparison with special handling
        word_similarities = []
        for q_word, db_word in zip(query_words, db_words):
            word_sim = self._calculate_word_similarity(q_word, db_word)
            word_similarities.append(word_sim)

        # Average similarity across all words
        return sum(word_similarities) / len(word_similarities) if word_similarities else 0.0

    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words with special handling for:
        - Gender variations (derecho/derecha)
        - Abbreviations (der/derecha, izq/izquierda)
        - Partial matches (dere/derecha)
        - Plurals and singulars (farola/farolas, paragolpe/paragolpes)

        CASE-INSENSITIVE: All comparisons are done in lowercase.
        """
        if not word1 or not word2:
            return 0.0

        # Convert to lowercase for case-insensitive comparison
        word1 = word1.lower()
        word2 = word2.lower()

        # Exact match
        if word1 == word2:
            return 1.0

        # Import the gender variant checker
        from utils.text_utils import are_gender_variants

        # Check for exact gender variants first (highest priority)
        if are_gender_variants(word1, word2):
            return 1.0  # Perfect match for gender variants

        # Handle common abbreviations and their full forms
        abbreviation_map = {
            'der': ['derecho', 'derecha'],
            'izq': ['izquierdo', 'izquierda'],
            'iz': ['izquierdo', 'izquierda'],
            'i': ['izquierdo', 'izquierda'],
            'del': ['delantero', 'delantera'],
            'delan': ['delantero', 'delantera'],
            'tra': ['trasero', 'trasera'],
            'tras': ['trasero', 'trasera'],
            't': ['trasero', 'trasera'],  # Single letter abbreviation
            'd': ['derecho', 'derecha', 'delantero', 'delantera'],  # Ambiguous - could be either
            'ant': ['anterior'],
            'post': ['posterior'],
            'sup': ['superior'],
            'inf': ['inferior'],
        }

        # Check if one word is an abbreviation of the other
        for abbrev, full_forms in abbreviation_map.items():
            if word1 == abbrev and word2 in full_forms:
                return 0.95  # High similarity for known abbreviations
            if word2 == abbrev and word1 in full_forms:
                return 0.95

        # Handle gender variations with pattern matching (fallback for unknown words)
        if len(word1) > 2 and len(word2) > 2:
            # Check if words are identical except for last character (gender)
            if word1[:-1] == word2[:-1] and word1[-1] in 'oa' and word2[-1] in 'oa':
                return 0.95  # High similarity for gender variations

        # Handle plurals and singulars
        plural_similarity = self._check_plural_singular_similarity(word1, word2)
        if plural_similarity > 0:
            return plural_similarity

        # Handle partial matches (one word is a prefix of the other)
        if word1.startswith(word2) or word2.startswith(word1):
            shorter = min(word1, word2, key=len)
            longer = max(word1, word2, key=len)
            if len(shorter) >= 3:  # Minimum 3 characters for partial match
                return 0.8 + 0.1 * (len(shorter) / len(longer))

        # Use sequence matching for other cases
        from difflib import SequenceMatcher
        return SequenceMatcher(None, word1, word2).ratio()

    def _check_plural_singular_similarity(self, word1: str, word2: str) -> float:
        """
        Check if two words are plural/singular variations of each other.
        Spanish plural rules:
        - Add 's' to words ending in vowel: farola → farolas
        - Add 'es' to words ending in consonant: paragolpe → paragolpes
        - Some irregular plurals
        """
        if len(word1) < 3 or len(word2) < 3:
            return 0.0

        # Check if one is plural of the other
        longer = max(word1, word2, key=len)
        shorter = min(word1, word2, key=len)

        # Rule 1: Add 's' (farola → farolas)
        if longer == shorter + 's':
            return 0.92

        # Rule 2: Add 'es' (paragolpe → paragolpes)
        if longer == shorter + 'es':
            return 0.92

        # Rule 3: Some words change ending (e.g., luz → luces)
        # Check if removing 'es' and adding common singular endings works
        if longer.endswith('es') and len(longer) > 3:
            stem = longer[:-2]  # Remove 'es'
            # Try common singular endings
            singular_candidates = [stem, stem + 'z', stem + 'x']
            if shorter in singular_candidates:
                return 0.92

        # Rule 4: Words ending in 'z' become 'ces' (luz → luces)
        if longer.endswith('ces') and shorter.endswith('z'):
            if longer[:-3] == shorter[:-1]:  # Compare stems
                return 0.92

        return 0.0

    def load_equivalencias_data(self, file_path: str) -> dict:
        """
        Load equivalencias data and create both ID mapping and synonym expansion dictionary.
        Returns the ID mapping (for backward compatibility).
        """
        print(f"Loading equivalencias from: {file_path}")
        if not os.path.exists(file_path):
            print(
                f"Warning: Equivalencias file not found at {file_path}. Equivalency linking will be disabled.")
            return {}
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            equivalencias_map = {}
            synonym_expansion_map = {}  # New: maps synonyms to equivalence group IDs

            for index, row in df.iterrows():
                equivalencia_row_id = index + 1
                row_terms = []  # Collect all terms in this row

                # First pass: collect all normalized terms in this row
                for col_name in df.columns:
                    term = row[col_name]
                    if pd.notna(term) and str(term).strip():
                        normalized_term = normalize_text(str(term))
                        if normalized_term:
                            row_terms.append(normalized_term)
                            # Store with lowercase key for case-insensitive lookup
                            equivalencias_map[normalized_term.lower()] = equivalencia_row_id

                # Second pass: create synonym mappings (all terms map to the equivalence group ID)
                # Ensure case-insensitive storage by using lowercase keys
                if row_terms:
                    # All terms in this row belong to the same equivalence group
                    for term in row_terms:
                        synonym_expansion_map[term.lower()] = equivalencia_row_id

            # Store the synonym expansion map globally for use in preprocessing
            global synonym_expansion_map_global
            synonym_expansion_map_global = synonym_expansion_map

            print(f"Loaded {len(equivalencias_map)} normalized term mappings from {len(df)} rows in Equivalencias.")
            print(f"Created {len(synonym_expansion_map)} synonym expansion mappings.")
            return equivalencias_map
        except Exception as e:
            print(
                f"Error loading or processing Equivalencias.xlsx: {e}. Equivalency linking will be disabled.")
            return {}

    def load_maestro_data(self, file_path: str, equivalencias_map: dict) -> list:
        # (Content remains the same as before)
        print(f"Loading maestro data from: {file_path}")
        # Updated column list - removed VIN_Model, VIN_BodyStyle, Equivalencia_Row_ID
        maestro_columns = [
            'Maestro_ID', 'VIN_Make', 'VIN_Year_Min', 'VIN_Year_Max',
            'VIN_Series_Trim', 'Original_Description_Input',
            'Normalized_Description_Input', 'Confirmed_SKU',
            'Confidence', 'Source', 'Date_Added'
        ]
        data_dir = os.path.dirname(file_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(file_path):
            print(
                f"Maestro file not found at {file_path}. Creating a new one with headers.")
            df_maestro = pd.DataFrame(columns=maestro_columns)
            try:
                df_maestro.to_excel(file_path, index=False)
                return []
            except Exception as e:
                print(
                    f"Error creating Maestro.xlsx: {e}. Maestro data will be empty.")
                return []
        try:
            df_maestro = pd.read_excel(file_path, sheet_name=0)
            if 'Maestro_ID' in df_maestro.columns:
                df_maestro['Maestro_ID'] = pd.to_numeric(
                    df_maestro['Maestro_ID'], errors='coerce').fillna(0).astype(int)
            if df_maestro.empty:
                print("Maestro file is empty.")
                return []
            maestro_list = []
            for _, row in df_maestro.iterrows():
                entry = row.to_dict()

                # Fix bracketed values in text columns
                for col in ['VIN_Make', 'VIN_Series_Trim']:
                    if col in entry and pd.notna(entry[col]):
                        value_str = str(entry[col]).strip()
                        if value_str.startswith("['") and value_str.endswith("']"):
                            entry[col] = value_str[2:-2]  # Remove [''] format
                        elif value_str.startswith('[') and value_str.endswith(']'):
                            entry[col] = value_str[1:-1].strip("'\"")  # Remove [] format

                original_desc = entry.get('Original_Description_Input', "")
                if pd.notna(original_desc):
                    normalized_desc = normalize_text(str(original_desc))
                    entry['Normalized_Description_Input'] = normalized_desc
                else:
                    entry['Normalized_Description_Input'] = ""

                # Fix bracketed values in integer columns (removed Equivalencia_Row_ID)
                for col in ['Maestro_ID', 'VIN_Year_Min', 'VIN_Year_Max']:
                    if col in entry and pd.notna(entry[col]):
                        original_value = entry[col]  # Store original value
                        try:
                            value_str = str(entry[col]).strip()
                            if value_str.startswith('[') and value_str.endswith(']'):
                                value_str = value_str[1:-1].strip("'\"")  # Remove [2012] -> 2012
                            entry[col] = int(value_str)
                        except (ValueError, TypeError):
                            # Keep original value if conversion fails, don't set to None
                            print(f"Warning: Could not convert {col} value '{original_value}' to integer, keeping original value")
                            entry[col] = original_value
                    # Don't set to None if column exists but is NaN - keep the original value
                if 'Confidence' in entry and pd.notna(entry['Confidence']):
                    try:
                        entry['Confidence'] = float(entry['Confidence'])
                    except (ValueError, TypeError):
                        entry['Confidence'] = None
                elif 'Confidence' in entry:
                    entry['Confidence'] = None
                maestro_list.append(entry)
            print(f"Loaded {len(maestro_list)} records from Maestro.xlsx.")
            return maestro_list
        except Exception as e:
            print(
                f"Error loading or processing Maestro.xlsx: {e}. Maestro data will be empty.")
            return []

    # Removed load_vin_details_from_chunks

    def create_widgets(self):
        """Creates the GUI widgets."""
        # --- Top Frame with two columns ---
        top_frame = ttk.Frame(self.root, padding=(10, 5))
        top_frame.pack(padx=10, pady=10, fill="x", expand=False)

        # Configure columns for 60/40 split
        top_frame.columnconfigure(0, weight=60)  # Left column (60%)
        top_frame.columnconfigure(1, weight=40)  # Right column (40%)

        # --- Input Frame (Left Column) ---
        input_frame = ttk.LabelFrame(top_frame, text="Input", padding=(10, 5))
        input_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")

        # VIN Input
        ttk.Label(input_frame, text="VIN (17 characters):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
        self.vin_entry = ttk.Entry(input_frame, width=25)
        self.vin_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Part Descriptions Input
        # Create a frame to hold the label and the instruction text
        part_desc_label_frame = ttk.Frame(input_frame)
        part_desc_label_frame.grid(
            row=1, column=0, padx=5, pady=5, sticky="nw")

        # Main label
        ttk.Label(part_desc_label_frame, text="Part Descriptions").grid(
            row=0, column=0, sticky="nw")

        # Instruction text below the main label
        ttk.Label(part_desc_label_frame, text="(one per line)", font=("", 8)).grid(
            row=1, column=0, sticky="nw")

        # Text input field - now spans across columns 1 and 2 to continue under the Find SKUs button
        self.parts_text = tk.Text(
            input_frame, width=40, height=5)  # Reduced width
        self.parts_text.grid(row=1, column=1, padx=5,
                             pady=5, sticky="ew", columnspan=2)
        input_frame.columnconfigure(1, weight=1)  # Allow parts_text to expand

        # Create a custom style for buttons with better contrast
        self.style = ttk.Style()
        self.style.configure("Accent.TButton",
                             background="#333333",  # Dark gray background
                             foreground="#ffffff",  # White text
                             font=("", 10, "bold"))  # Bold font

        # Add a style for the manual entry confirm button
        self.style.configure("Manual.TButton",
                             background="#333333",  # Dark gray background
                             foreground="#ffffff",  # White text
                             font=("", 9, "bold"))  # Bold font

        # Create a custom button class that uses a dark background with white text
        class DarkButton(tk.Button):
            def __init__(self, master=None, **kwargs):
                super().__init__(master, **kwargs)
                self.configure(
                    background="#333333",  # Dark gray background
                    foreground="#ffffff",  # White text
                    font=("", 10, "bold"),  # Bold font
                    borderwidth=1,
                    relief=tk.RAISED,
                    padx=10,
                    pady=5,
                    activebackground="#555555",  # Slightly lighter when clicked
                    activeforeground="#ffffff",  # White text when active
                    disabledforeground="#ffffff"  # White text when disabled
                )

        # Find SKUs Button - moved to the position indicated by the red arrow (between VIN and Part Descriptions)
        self.find_button = DarkButton(
            input_frame, text="Find SKUs",
            command=self.find_skus_handler)
        self.find_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # --- Vehicle Details Frame (Right Column) ---
        self.vehicle_details_frame = ttk.LabelFrame(
            top_frame, text="Predicted Vehicle Details", padding=(10, 5))
        self.vehicle_details_frame.grid(
            row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")

        # Placeholder for vehicle details (will be populated after prediction)
        self.vehicle_details_placeholder = ttk.Label(
            self.vehicle_details_frame, text="Vehicle details will appear here after VIN prediction.")
        self.vehicle_details_placeholder.pack(padx=5, pady=5, anchor="nw")

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(
            # Changed title to be more specific
            self.root, text="SKU Suggestions", padding=(10, 5))
        output_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Make the output frame resize correctly
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        # Scrollable Canvas for results
        canvas = tk.Canvas(output_frame)
        scrollbar = ttk.Scrollbar(
            output_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        # Configure the scrollable frame to expand to fill the canvas width
        self.scrollable_frame.columnconfigure(0, weight=1)

        # Bind the frame to update the scrollregion when its size changes
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Create window with scrollable frame and configure it to expand horizontally
        canvas.create_window((0, 0), window=self.scrollable_frame,
                             anchor="nw", width=canvas.winfo_width())
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make sure the canvas expands to fill the frame
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Update the canvas when it's resized to adjust the scrollable frame width
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        # Placeholder Label inside scrollable frame
        self.results_placeholder_label = ttk.Label(
            self.scrollable_frame, text="Enter VIN and Descriptions, then click 'Find SKUs'.")
        self.results_placeholder_label.pack(padx=5, pady=5, anchor="nw")

        # --- Bottom Frame for Save Button ---
        bottom_frame = ttk.Frame(self.root, padding=(10, 5))
        bottom_frame.pack(side=tk.BOTTOM, fill="x",
                          expand=False, pady=(0, 10), padx=10)

        self.save_button = DarkButton(
            bottom_frame, text="Save Confirmed Selections",
            command=self.save_selections_handler,
            state=tk.DISABLED)
        # Pack inside the bottom_frame with some padding
        self.save_button.pack(pady=5)

        # Instance variables
        self.vehicle_details = None
        self.processed_parts = None
        self.current_suggestions = {}
        self.selection_vars = {}
        self.part_frames_widgets = []  # To store part_frame widgets for responsive layout
        self.current_num_columns = 0  # To track current number of columns in results

        # Removed manual input variables

    def _correct_vin(self, vin: str) -> str:
        """
        Corrects common VIN input mistakes:
        - Replaces I/i with 1
        - Replaces O/o/Q/q with 0
        Returns the corrected VIN string.
        """
        return vin.upper().replace('I', '1').replace('O', '0').replace('Q', '0')

    def _format_confidence_percentage(self, confidence: float) -> str:
        """
        Converts confidence score to percentage format for display.

        Args:
            confidence: Confidence score (0.0-1.0)

        Returns:
            Formatted percentage string (e.g., "75%")
        """
        return f"{int(confidence * 100)}%"

    def _get_sku_nn_prediction(self, make: str, model_year: str, series: str, description: str) -> str | None:
        """
        Uses the loaded SKU NN model to predict an SKU.
        Returns the predicted SKU string or None if prediction fails or model not available.
        """
        if not sku_nn_model or not sku_nn_encoder_make or not sku_nn_encoder_model_year or \
           not sku_nn_encoder_series or not sku_nn_tokenizer_desc or not sku_nn_encoder_sku:
            print(
                "SKU NN model or one of its preprocessors is not loaded. Skipping NN prediction.")
            return None

        try:
            # Use our predict_sku function from sku_nn_pytorch.py
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Create a dictionary of encoders to pass to predict_sku
            encoders = {
                'Make': sku_nn_encoder_make,
                'Model Year': sku_nn_encoder_model_year,
                'Series': sku_nn_encoder_series,
                'tokenizer': sku_nn_tokenizer_desc,
                'sku': sku_nn_encoder_sku
            }

            # Call the predict_sku function
            predicted_sku, confidence = predict_sku(
                model=sku_nn_model,
                encoders=encoders,
                make=make,
                model_year=model_year,
                series=series,
                description=description,
                device=device
            )

            if predicted_sku and predicted_sku.strip():
                print(
                    f"  SKU NN Prediction for '{description}': {predicted_sku} (Confidence: {confidence:.4f})")
                return predicted_sku, confidence
            else:
                print(
                    f"  SKU NN Prediction failed for '{description}' or returned empty SKU")
                return None

        except ValueError as ve:
            # This can happen if a category (Make, Year, Series) was not seen during training
            print(
                f"  SKU NN Prediction Error: Could not encode inputs for '{description}'. Untrained category? Details: {ve}")
            return None
        except Exception as e:
            print(f"  Error during SKU NN prediction for '{description}': {e}")
            return None

    def find_skus_handler(self):
        """
        Handles the 'Find SKUs' button click.
        Uses trained models to predict VIN details, processes parts, searches, and displays.
        """
        print("\n--- 'Find SKUs' button clicked ---")
        vin = self.vin_entry.get().strip().upper()
        # VIN correction step
        corrected_vin = self._correct_vin(vin)
        if vin != corrected_vin:
            print(f"VIN corrected from {vin} to {corrected_vin}")
            self.vin_entry.delete(0, 'end')
            self.vin_entry.insert(0, corrected_vin)
        vin = corrected_vin

        # Clear previous results
        self._clear_results_area()

        # Clear vehicle details frame
        for widget in self.vehicle_details_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.vehicle_details_frame,
                  text="Processing VIN...").pack(anchor="w", padx=5, pady=5)

        print(f"VIN Entered: {vin}")

        # Validate VIN format
        if not vin or len(vin) != 17 or not re.match("^[A-HJ-NPR-Z0-9]{17}$", vin):
            messagebox.showerror(
                "Invalid VIN", "VIN must be 17 alphanumeric characters (excluding I, O, Q).")
            ttk.Label(self.scrollable_frame,
                      text="Error: Invalid VIN format.").pack(anchor="nw")
            return

        # --- Predict VIN Details using Models ---
        predicted_details = self.predict_vin_details(vin)

        if not predicted_details:
            ttk.Label(self.scrollable_frame,
                      text=f"Could not predict details for VIN: {vin}.").pack(anchor="nw")
            # Optionally allow manual input here if prediction fails completely
            # self._prompt_for_manual_details(vin) # If we want fallback to manual
            return

        self.vehicle_details = predicted_details  # Store predicted details
        print(f"Predicted Vehicle Details: {self.vehicle_details}")

        # Proceed with part processing and search using predicted details
        self._process_parts_and_continue_search()

    def predict_vin_details(self, vin: str) -> dict | None:
        """Predicts Make, Year, Series using loaded models."""
        if not model_maker or not model_year or not model_series:
            print("Error: Prediction models not loaded.")
            messagebox.showerror(
                "Prediction Error", "VIN prediction models are not loaded. Cannot proceed.")
            return None

        features = extract_vin_features(vin)
        if not features:
            messagebox.showerror("Prediction Error",
                                 "Could not extract features from VIN.")
            return None

        details = {"Make": "N/A", "Model Year": "N/A",
                   "Series": "N/A", "Model": "N/A", "Body Class": "N/A"}

        try:
            # Predict Maker - Use DataFrame with proper column names
            import pandas as pd
            wmi_df = pd.DataFrame([[features['wmi']]], columns=['wmi'])
            wmi_encoded = encoder_x_maker.transform(wmi_df)
            # Check for unknown category before prediction
            if -1 in wmi_encoded:
                details['Make'] = "Unknown (WMI)"
            else:
                maker_pred_encoded = model_maker.predict(wmi_encoded)
                # Check for unknown category in prediction output (shouldn't happen with CategoricalNB if input known)
                if maker_pred_encoded[0] != -1:
                    make_result = encoder_y_maker.inverse_transform(
                        maker_pred_encoded.reshape(-1, 1))[0]
                    # Ensure it's a scalar value, not an array
                    if hasattr(make_result, 'item'):
                        details['Make'] = make_result.item()
                    else:
                        details['Make'] = make_result
                else:  # Should not happen if input was known, but handle defensively
                    details['Make'] = "Unknown (Prediction)"

            # Predict Year - Use DataFrame with proper column names
            year_df = pd.DataFrame([[features['year_code']]], columns=['year_code'])
            year_code_encoded = encoder_x_year.transform(year_df)
            if -1 in year_code_encoded:
                details['Model Year'] = "Unknown (Code)"
            else:
                year_pred_encoded = model_year.predict(year_code_encoded)
                if year_pred_encoded[0] != -1:
                    year_result = encoder_y_year.inverse_transform(
                        year_pred_encoded.reshape(-1, 1))[0]
                    # Ensure it's a scalar value, not an array
                    if hasattr(year_result, 'item'):
                        details['Model Year'] = year_result.item()
                    else:
                        details['Model Year'] = year_result
                else:
                    # Fallback to direct map if model fails (unlikely for year)
                    details['Model Year'] = str(decode_year(features['year_code'])) if decode_year(
                        features['year_code']) else "Unknown (Code)"

            # Predict Series - Use DataFrame with proper column names
            series_df = pd.DataFrame([[features['wmi'], features['vds_full']]],
                                   columns=['wmi', 'vds_full'])
            series_features_encoded = encoder_x_series.transform(series_df)
            # Check if either feature was unknown
            if -1 in series_features_encoded[0]:
                details['Series'] = "Unknown (VDS/WMI)"
            else:
                series_pred_encoded = model_series.predict(
                    series_features_encoded)
                if series_pred_encoded[0] != -1:
                    series_result = encoder_y_series.inverse_transform(
                        series_pred_encoded.reshape(-1, 1))[0]
                    # Ensure it's a scalar value, not an array
                    if hasattr(series_result, 'item'):
                        details['Series'] = series_result.item()
                    else:
                        details['Series'] = series_result
                else:
                    details['Series'] = "Unknown (Prediction)"

            # Model and Body Class are not predicted by these models
            details['Model'] = "N/A (Not Predicted)"
            details['Body Class'] = "N/A (Not Predicted)"

            return details

        except Exception as e:
            print(f"Error during VIN prediction: {e}")
            # Handle potential errors if a feature wasn't seen during training
            messagebox.showwarning(
                "Prediction Warning", f"Could not reliably predict all details for VIN: {e}")
            # Return partially filled details if possible
            return details

    def _clear_results_area(self):
        """Clears the widgets in the scrollable results frame."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        # Removed manual input widget clearing
        self.results_placeholder_label = None
        self.save_button.config(state=tk.DISABLED)

    # Removed _prompt_for_manual_details
    # Removed _handle_manual_details_continue

    def _is_valid_sku(self, sku: str) -> bool:
        """
        Validates if a SKU is acceptable for suggestions.
        Filters out UNKNOWN, empty, or invalid SKUs.
        """
        if not sku or not sku.strip():
            return False

        # Convert to uppercase for consistent checking
        sku_upper = sku.strip().upper()

        # Filter out UNKNOWN and similar invalid values
        invalid_skus = {'UNKNOWN', 'N/A', 'NULL', 'NONE', '', 'TBD', 'PENDING', 'MANUAL'}

        if sku_upper in invalid_skus:
            print(f"    Filtered out invalid SKU: '{sku}'")
            return False

        return True

    def _aggregate_sku_suggestions(self, suggestions: dict, new_sku: str, new_confidence: float, new_source: str) -> dict:
        """
        Aggregates SKU suggestions, handling duplicates by keeping the highest confidence.
        Also tracks all sources for transparency and applies consensus-based confidence adjustment.

        Confidence Rules:
        - Single source (Maestro only): Max 0.90 confidence
        - Single source (NN only): Max 0.85 confidence
        - Single source (DB only): Max 0.70 confidence
        - Multiple sources (Maestro + NN): Can reach 1.00 confidence
        - Multiple sources (any combination): Bonus for consensus
        """
        if not self._is_valid_sku(new_sku):
            return suggestions

        if new_sku in suggestions:
            # SKU already exists - aggregate information
            existing = suggestions[new_sku]
            existing_conf = existing["confidence"]
            existing_source = existing["source"]
            existing_all_sources = existing.get("all_sources", existing_source)

            # Track all sources
            all_sources_list = existing_all_sources.split(", ") if existing_all_sources else [existing_source]
            if new_source not in all_sources_list:
                all_sources_list.append(new_source)

            combined_sources = ", ".join(all_sources_list)

            # Calculate consensus-adjusted confidence
            adjusted_confidence = self._calculate_consensus_confidence(
                max(existing_conf, new_confidence), all_sources_list
            )

            # Update with consensus-adjusted confidence
            suggestions[new_sku] = {
                "confidence": adjusted_confidence,
                "source": new_source if new_confidence > existing_conf else existing["source"],
                "all_sources": combined_sources,
                "best_confidence": max(existing_conf, new_confidence),
                "source_count": len(all_sources_list)
            }
            print(f"    🔄 Consensus update {new_sku}: {max(existing_conf, new_confidence):.3f} -> {adjusted_confidence:.3f} (Sources: {combined_sources})")
        else:
            # New SKU - add it with single-source confidence adjustment
            adjusted_confidence = self._calculate_consensus_confidence(new_confidence, [new_source])
            suggestions[new_sku] = {
                "confidence": adjusted_confidence,
                "source": new_source,
                "all_sources": new_source,
                "best_confidence": new_confidence,
                "source_count": 1
            }
            if adjusted_confidence != new_confidence:
                print(f"    📉 Single-source adjustment {new_sku}: {new_confidence:.3f} -> {adjusted_confidence:.3f} ({new_source})")

        return suggestions

    def _calculate_consensus_confidence(self, base_confidence: float, sources: list) -> float:
        """
        Calculate consensus-adjusted confidence based on prediction sources.

        Updated rules for new priority system:
        - Single source (Maestro only): Max 90% (0.90) confidence
        - Single source (NN only): Max 85% (0.85) confidence
        - Single source (DB only): Max 80% (0.80) confidence
        - NN + DB consensus: Higher value + 10% boost
        - Maestro + NN consensus: 100% (1.0) confidence
        - All three sources: 100% (1.0) confidence

        Args:
            base_confidence: Original confidence score
            sources: List of prediction sources

        Returns:
            Adjusted confidence score
        """
        if len(sources) == 1:
            # Single source - apply caps
            source = sources[0]
            if "Maestro" in source:
                return min(base_confidence, 0.90)  # 90% max for Maestro alone
            elif "SKU-NN" in source or "NN" in source:
                return min(base_confidence, 0.85)  # 85% max for NN alone
            elif "DB" in source:
                return min(base_confidence, 0.80)  # 80% max for DB alone (increased from 70%)
            else:
                return min(base_confidence, 0.80)  # Default single source cap

        else:
            # Multiple sources - consensus bonus
            has_maestro = any("Maestro" in source for source in sources)
            has_nn = any("SKU-NN" in source or "NN" in source for source in sources)
            has_db = any("DB" in source for source in sources)

            # All three sources = 100% confidence
            if has_maestro and has_nn and has_db:
                return 1.00

            # Maestro + NN consensus = 100% confidence
            elif has_maestro and has_nn:
                return 1.00

            # NN + DB consensus: Higher value + 10% boost
            elif has_nn and has_db:
                return min(base_confidence + 0.10, 0.95)

            # Other multi-source combinations get smaller bonus
            elif len(sources) >= 2:
                return min(base_confidence + 0.05, 0.90)

            return base_confidence

    def _process_parts_and_continue_search(self):
        """Processes part descriptions and triggers the search and display."""
        # (Content remains largely the same, uses self.vehicle_details which is now predicted)
        part_descriptions_raw = self.parts_text.get("1.0", tk.END).strip()
        self.processed_parts = []
        if part_descriptions_raw:
            original_descriptions = [
                line.strip() for line in part_descriptions_raw.splitlines() if line.strip()]
            print(f"Original Descriptions List: {original_descriptions}")
            for original_desc in original_descriptions:
                print(f"  Processing: '{original_desc}'")

                # STEP 1: Normalize the original description (without synonym expansion)
                normalized_original = normalize_text(original_desc)
                print(f"  Normalized original: '{normalized_original}'")

                # STEP 2: Apply synonym expansion for fallback searches
                expanded_desc = self.expand_synonyms(original_desc)
                print(f"  After synonym expansion: '{expanded_desc}'")
                normalized_expanded = normalize_text(expanded_desc)
                print(f"  Normalized expanded: '{normalized_expanded}'")

                # STEP 3: Create abbreviated version to match database format
                abbreviated_desc = self.create_abbreviated_version(normalized_original)
                print(f"  Abbreviated version: '{abbreviated_desc}'")

                # STEP 3: Look up equivalencia ID using the expanded form (CASE-INSENSITIVE)
                equivalencia_id = equivalencias_map_global.get(normalized_expanded.lower())

                # STEP 4: If no direct match, try fuzzy matching as fallback
                if equivalencia_id is None:
                    fuzzy_normalized_desc = normalize_text(expanded_desc, use_fuzzy=True)
                    equivalencia_id = equivalencias_map_global.get(fuzzy_normalized_desc.lower())

                    if equivalencia_id is not None:
                        normalized_expanded = fuzzy_normalized_desc
                        print(f"  Found via fuzzy normalization: EqID {equivalencia_id}")

                    # Final fallback: try fuzzy matching against all equivalencias terms
                    if equivalencia_id is None and equivalencias_map_global:
                        normalized_terms = list(equivalencias_map_global.keys())
                        try:
                            from utils.fuzzy_matcher import find_best_match
                            match_result = find_best_match(
                                normalized_expanded, normalized_terms, threshold=0.8)

                            if match_result:
                                best_match, similarity = match_result
                                equivalencia_id = equivalencias_map_global.get(best_match.lower())
                                normalized_expanded = best_match
                                print(f"  Found via fuzzy match: '{best_match}' (similarity: {similarity:.2f}, EqID: {equivalencia_id})")
                        except ImportError:
                            pass

                self.processed_parts.append({
                    "original": original_desc,
                    "normalized_original": normalized_original,  # Store original normalized form
                    "expanded": expanded_desc,  # Store the synonym-expanded form
                    "normalized_expanded": normalized_expanded,  # Store expanded normalized form
                    "abbreviated": abbreviated_desc,  # Store abbreviated form for database matching
                    "equivalencia_id": equivalencia_id
                })
                print(f"  Final result: '{original_desc}' -> Original normalized: '{normalized_original}' -> Abbreviated: '{abbreviated_desc}' -> Expanded: '{expanded_desc}' -> Expanded normalized: '{normalized_expanded}', EqID: {equivalencia_id}")
        else:
            print("No part descriptions entered.")
            self.processed_parts = []

        # --- Phase 4: Search Logic ---
        self.current_suggestions = {}
        self.selection_vars = {}

        db_conn = None
        try:
            print(f"Connecting to database: {DEFAULT_DB_PATH}")
            if not os.path.exists(DEFAULT_DB_PATH):
                messagebox.showerror(
                    "Database Error", f"Database file not found at {DEFAULT_DB_PATH}. Please run the offline processor first.")
                ttk.Label(self.scrollable_frame, text=f"Error: Database not found at {DEFAULT_DB_PATH}").pack(
                    anchor="nw")
                return
            db_conn = sqlite3.connect(DEFAULT_DB_PATH)
            cursor = db_conn.cursor()
            print("Database connection successful.")

            for part_info in self.processed_parts:
                original_desc = part_info["original"]
                normalized_original = part_info["normalized_original"]
                expanded_desc = part_info["expanded"]
                normalized_expanded = part_info["normalized_expanded"]
                abbreviated_desc = part_info["abbreviated"]
                eq_id = part_info["equivalencia_id"]
                print(
                    f"\nSearching for: '{original_desc}' (Original normalized: '{normalized_original}', Abbreviated: '{abbreviated_desc}', Expanded: '{expanded_desc}', Expanded normalized: '{normalized_expanded}', EqID: {eq_id})")

                suggestions = {}

                # Use self.vehicle_details which is now PREDICTED
                predicted_make_val = self.vehicle_details.get('Make', 'N/A')
                if isinstance(predicted_make_val, np.ndarray):
                    vin_make_raw = str(predicted_make_val.item()) if predicted_make_val.size > 0 else 'N/A'
                else:
                    vin_make_raw = str(predicted_make_val) if pd.notna(predicted_make_val) else 'N/A'

                # Fix make case - database AND Maestro use proper case, not uppercase
                # Based on database analysis: 'Renault', 'Chevrolet', 'Ford', 'Mazda', etc.
                make_case_map = {
                    'RENAULT': 'Renault',
                    'CHEVROLET': 'Chevrolet',
                    'MAZDA': 'Mazda',
                    'FORD': 'Ford',
                    'HYUNDAI': 'Hyundai',
                    'TOYOTA': 'Toyota',
                    'NISSAN': 'Nissan',
                    'KIA': 'Kia',
                    'VOLKSWAGEN': 'Volkswagen'
                }
                vin_make = make_case_map.get(vin_make_raw.upper(), vin_make_raw)
                print(f"  🔧 Make case correction: '{vin_make_raw}' -> '{vin_make}'")
                print(f"  Make case correction: '{vin_make_raw}' -> '{vin_make}'")

                # Model is likely N/A from predictor
                vin_model = self.vehicle_details.get('Model', 'N/A')
                vin_year_str = self.vehicle_details.get('Model Year', 'N/A')
                vin_year = None  # Initialize
                # Check if it's still an array element
                if isinstance(vin_year_str, np.ndarray):
                    vin_year_str_scalar = vin_year_str.item()  # Extract scalar value
                else:
                    vin_year_str_scalar = vin_year_str  # Use as is if already scalar/string

                if vin_year_str_scalar and vin_year_str_scalar != 'N/A':
                    try:
                        vin_year = int(vin_year_str_scalar)
                    except (ValueError, TypeError):
                        print(
                            f"Warning: Could not convert predicted year '{vin_year_str_scalar}' to integer.")
                        vin_year = None  # Ensure it's None if conversion fails

                # Enhanced Maestro Search with 3-parameter exact + fuzzy description matching
                print(f"  Searching Maestro data ({len(maestro_data_global)} entries)...")

                # Get predicted series for matching
                predicted_series_val = self.vehicle_details.get('Series', 'N/A')
                if isinstance(predicted_series_val, np.ndarray):
                    vin_series = str(predicted_series_val.item()) if predicted_series_val.size > 0 else 'N/A'
                else:
                    vin_series = str(predicted_series_val) if pd.notna(predicted_series_val) else 'N/A'

                # Debug: Show what we're searching for
                print(f"    🔍 Searching for: Make='{vin_make}', Year='{vin_year_str_scalar}', Series='{vin_series}'")
                print(f"    🔍 Description (original): '{normalized_original}'")
                print(f"    🔍 Description (expanded): '{normalized_expanded}'")

                # First pass: Exact matches on Make, Year, Series + exact description
                maestro_matches_found = 0
                for maestro_entry in maestro_data_global:
                    maestro_make = str(maestro_entry.get('VIN_Make', ''))
                    maestro_year = str(maestro_entry.get('VIN_Year_Min', ''))
                    maestro_series = str(maestro_entry.get('VIN_Series_Trim', ''))
                    maestro_desc = str(maestro_entry.get('Normalized_Description_Input', '')).lower()

                    # Check 3-parameter exact match (Make, Year, Series) - CASE INSENSITIVE
                    make_match = maestro_make.upper() == vin_make.upper()
                    year_match = maestro_year == vin_year_str_scalar
                    series_match = maestro_series.upper() == vin_series.upper()

                    # Debug: Show first few comparisons
                    if maestro_matches_found < 3:
                        print(f"    📋 Maestro entry: Make='{maestro_make.upper()}' vs '{vin_make.upper()}' ({make_match})")
                        print(f"    📋 Maestro entry: Year='{maestro_year}' vs '{vin_year_str_scalar}' ({year_match})")
                        print(f"    📋 Maestro entry: Series='{maestro_series.upper()}' vs '{vin_series.upper()}' ({series_match})")
                        maestro_matches_found += 1

                    if make_match and year_match and series_match:
                        # Apply unified preprocessing for exact description matching
                        preprocessed_maestro_desc = self.unified_text_preprocessing(maestro_desc)
                        preprocessed_original = self.unified_text_preprocessing(original_desc)
                        preprocessed_expanded = self.unified_text_preprocessing(expanded_desc)

                        # Check for exact description match after unified preprocessing
                        desc_match_orig = preprocessed_maestro_desc == preprocessed_original
                        desc_match_exp = preprocessed_maestro_desc == preprocessed_expanded
                        desc_match = desc_match_orig or desc_match_exp

                        if desc_match:
                            sku = maestro_entry.get('Confirmed_SKU')
                            if sku and sku.strip():
                                suggestions = self._aggregate_sku_suggestions(
                                    suggestions, sku, 1.0, "Maestro (Unified Exact)")
                                print(f"    ✅ Found in Maestro (Unified Exact 4-param): {sku} (Conf: 1.0)")
                                match_type = "original" if desc_match_orig else "expanded"
                                print(f"      Matched via {match_type}: '{preprocessed_maestro_desc}'")

                # Second pass: Fuzzy description matching for same Make, Year, Series
                if not suggestions:  # Only do fuzzy if no exact matches found
                    try:
                        from utils.fuzzy_matcher import find_best_match

                        # Get all descriptions from matching Make, Year, Series entries
                        matching_entries = []
                        for maestro_entry in maestro_data_global:
                            maestro_make = str(maestro_entry.get('VIN_Make', ''))
                            maestro_year = str(maestro_entry.get('VIN_Year_Min', ''))
                            maestro_series = str(maestro_entry.get('VIN_Series_Trim', ''))

                            if (maestro_make.upper() == vin_make.upper() and
                                maestro_year == vin_year_str_scalar and
                                maestro_series.upper() == vin_series.upper()):
                                matching_entries.append(maestro_entry)

                        if matching_entries:
                            print(f"    🔄 Applying unified preprocessing for Maestro fuzzy matching...")

                            # Apply unified preprocessing to input descriptions
                            preprocessed_original = self.unified_text_preprocessing(original_desc)
                            preprocessed_expanded = self.unified_text_preprocessing(expanded_desc)

                            # Apply unified preprocessing to candidate descriptions and create mapping
                            candidate_descriptions = []
                            desc_to_entry_map = {}

                            for entry in matching_entries:
                                raw_desc = str(entry.get('Normalized_Description_Input', ''))
                                preprocessed_desc = self.unified_text_preprocessing(raw_desc)
                                candidate_descriptions.append(preprocessed_desc)
                                desc_to_entry_map[preprocessed_desc] = entry

                            print(f"    🔍 Comparing preprocessed input: '{preprocessed_original}' vs candidates")

                            # Find best fuzzy match (try original first, then expanded)
                            match_result = find_best_match(
                                preprocessed_original, candidate_descriptions, threshold=0.8)

                            # If no good match with original, try expanded
                            if not match_result or match_result[1] < 0.8:
                                print(f"    🔍 Trying expanded preprocessed input: '{preprocessed_expanded}'")
                                match_result_exp = find_best_match(
                                    preprocessed_expanded, candidate_descriptions, threshold=0.7)
                                if match_result_exp and (not match_result or match_result_exp[1] > match_result[1]):
                                    match_result = match_result_exp

                            if match_result:
                                best_match_desc, similarity = match_result
                                print(f"    ✅ Best match found: '{best_match_desc}' (Similarity: {similarity:.3f})")

                                # Find the entry with this preprocessed description
                                if best_match_desc in desc_to_entry_map:
                                    entry = desc_to_entry_map[best_match_desc]
                                    sku = entry.get('Confirmed_SKU')
                                    if sku and sku.strip():
                                        # Higher confidence for unified preprocessing matches
                                        confidence = round(0.8 + 0.2 * similarity, 3)
                                        suggestions = self._aggregate_sku_suggestions(
                                            suggestions, sku, confidence, f"Maestro (Unified Fuzzy: {similarity:.2f})")
                                        print(f"    ✅ Found in Maestro (Unified Fuzzy): {sku} (Sim: {similarity:.2f}, Conf: {confidence})")
                            else:
                                print(f"    ❌ No suitable fuzzy match found in Maestro after unified preprocessing")
                    except ImportError:
                        print("    Fuzzy matching not available for Maestro search")

                # --- Neural Network Prediction (Priority 2) ---
                # Ensure vehicle details are strings for the NN model
                predicted_series_val = self.vehicle_details.get('Series', 'N/A')
                if isinstance(predicted_series_val, np.ndarray) and predicted_series_val.size > 0:
                    vin_series_str_for_nn = str(predicted_series_val.item())
                elif pd.notna(predicted_series_val):
                    vin_series_str_for_nn = str(predicted_series_val)
                else:
                    vin_series_str_for_nn = "N/A"

                if vin_make != 'N/A' and vin_year_str_scalar != 'N/A' and vin_series_str_for_nn != 'N/A':
                    print(f"  🔄 Applying unified preprocessing for Neural Network input...")

                    # Apply unified preprocessing to input descriptions for Neural Network
                    preprocessed_original = self.unified_text_preprocessing(original_desc)
                    preprocessed_expanded = self.unified_text_preprocessing(expanded_desc)

                    print(f"  Attempting SKU NN prediction for: Make='{vin_make}', Year='{vin_year_str_scalar}', Series='{vin_series_str_for_nn}'")
                    print(f"  NN Input (preprocessed original): '{preprocessed_original}'")
                    print(f"  NN Input (preprocessed expanded): '{preprocessed_expanded}'")

                    # Try preprocessed original description first
                    sku_nn_output = self._get_sku_nn_prediction(
                        make=vin_make,
                        model_year=vin_year_str_scalar,
                        series=vin_series_str_for_nn,
                        description=preprocessed_original
                    )

                    # If no good result, try preprocessed expanded description
                    if not sku_nn_output or (sku_nn_output and sku_nn_output[1] < 0.7):
                        print(f"  Trying NN prediction with preprocessed expanded description...")
                        sku_nn_output_expanded = self._get_sku_nn_prediction(
                            make=vin_make,
                            model_year=vin_year_str_scalar,
                            series=vin_series_str_for_nn,
                            description=preprocessed_expanded
                        )
                        # Use expanded result if it's better
                        if sku_nn_output_expanded and (not sku_nn_output or sku_nn_output_expanded[1] > sku_nn_output[1]):
                            sku_nn_output = sku_nn_output_expanded

                    if sku_nn_output:
                        nn_sku, nn_confidence = sku_nn_output
                        if nn_sku and nn_sku.strip():  # Add if not empty
                            suggestions = self._aggregate_sku_suggestions(
                                suggestions, nn_sku, float(nn_confidence), "SKU-NN")
                            print(f"    Found via SKU-NN: {nn_sku} (Conf: {nn_confidence:.4f})")
                else:
                    print("  Skipping SKU NN prediction due to missing Make, Year, or Series from VIN prediction.")
                # --- End Neural Network Prediction ---

                # SQLite Search (4-parameter matching: Make, Year, Series, Description) - Priority 3
                if vin_year is not None and vin_series != 'N/A':
                    print(
                        f"  Searching SQLite DB (Make: {vin_make}, Year: {vin_year}, Series: {vin_series})...")
                    try:
                        # DUAL MATCHING STRATEGY: Try exact match first, then normalized match

                        # STEP 1A: Try exact description match (no normalization) - handles system-generated descriptions
                        print(f"    Trying exact match with original description: '{original_desc}'")
                        cursor.execute("""
                            SELECT sku, COUNT(*) as frequency
                            FROM historical_parts
                            WHERE vin_make = ? AND vin_year = ? AND vin_series = ? AND normalized_description = ?
                            GROUP BY sku
                            ORDER BY COUNT(*) DESC
                        """, (vin_make, vin_year, vin_series, original_desc))
                        exact_results = cursor.fetchall()

                        if exact_results:
                            print(f"    ✅ Found {len(exact_results)} unique SKUs via EXACT match")
                            # Apply consensus logic to filter out minority/outlier SKUs
                            consensus_skus = self.apply_consensus_logic(exact_results, min_consensus_ratio=0.6)
                            for sku, frequency in consensus_skus:
                                confidence = self.calculate_frequency_based_confidence(frequency, "DB-Exact")
                                final_confidence = self._calculate_consensus_confidence(confidence, ["DB-Exact"])
                                suggestions = self._aggregate_sku_suggestions(suggestions, sku, final_confidence, "DB-Exact")
                                print(f"    ✅ Found in DB (Exact Match): {sku} (Freq: {frequency}, Conf: {final_confidence:.4f})")

                        # STEP 1B: If no exact matches, try normalized abbreviated description
                        if not exact_results:
                            print(f"    No exact matches, trying normalized abbreviated: '{abbreviated_desc}'")
                            cursor.execute("""
                                SELECT sku, COUNT(*) as frequency
                                FROM historical_parts
                                WHERE vin_make = ? AND vin_year = ? AND vin_series = ? AND normalized_description = ?
                                GROUP BY sku
                                ORDER BY COUNT(*) DESC
                            """, (vin_make, vin_year, vin_series, abbreviated_desc))
                            results = cursor.fetchall()

                        if results:
                            print(f"    Found {len(results)} unique SKUs in DB (4-param Exact)")

                            # Apply consensus logic to filter out minority/outlier SKUs
                            consensus_skus = self.apply_consensus_logic(results, min_consensus_ratio=0.6)

                            # Process consensus SKUs with frequency-based confidence
                            for sku, frequency in consensus_skus:
                                if sku and sku.strip():
                                    confidence = self.calculate_frequency_based_confidence(frequency, "DB (4-param Exact)")
                                    suggestions = self._aggregate_sku_suggestions(
                                        suggestions, sku, confidence, f"DB (4-param Exact)")
                                    print(
                                        f"    ✅ Found in DB (4-param Exact): {sku} (Freq: {frequency}, Conf: {confidence})")
                        else:
                            print("    No exact matches found in DB (4-param)")

                        # STEP 2: If no exact match, try fuzzy series matching with abbreviated description
                        if not suggestions:
                            print("    No exact series match, trying fuzzy series matching with abbreviated description...")
                            cursor.execute("""
                                SELECT sku, COUNT(*) as frequency
                                FROM historical_parts
                                WHERE vin_make = ? AND vin_year = ? AND vin_series LIKE ? AND normalized_description = ?
                                GROUP BY sku
                                ORDER BY COUNT(*) DESC
                            """, (vin_make, vin_year, f'%{vin_series}%', abbreviated_desc))
                            results = cursor.fetchall()

                            if results:
                                print(f"    Found {len(results)} unique SKUs in DB (Fuzzy Series + Abbreviated)")

                                # Apply consensus logic to filter out minority/outlier SKUs
                                consensus_skus = self.apply_consensus_logic(results, min_consensus_ratio=0.6)

                                # Process consensus SKUs with frequency-based confidence
                                for sku, frequency in consensus_skus:
                                    if sku and sku.strip():
                                        confidence = self.calculate_frequency_based_confidence(frequency, "DB (Fuzzy Series)")
                                        suggestions = self._aggregate_sku_suggestions(
                                            suggestions, sku, confidence, f"DB (Fuzzy Series + Abbreviated)")
                                        print(
                                            f"    ✅ Found in DB (Fuzzy Series + Abbreviated): {sku} (Freq: {frequency}, Conf: {confidence})")
                            else:
                                print("    No fuzzy series + abbreviated matches found in DB")

                        # STEP 2b: If still no match, try fuzzy series matching with original description
                        if not suggestions:
                            print("    No match with abbreviated, trying fuzzy series matching with original description...")
                            cursor.execute("""
                                SELECT sku, COUNT(*) as frequency
                                FROM historical_parts
                                WHERE vin_make = ? AND vin_year = ? AND vin_series LIKE ? AND normalized_description = ?
                                GROUP BY sku
                            """, (vin_make, vin_year, f'%{vin_series}%', normalized_original))
                            results = cursor.fetchall()
                            total_matches = sum(row[1] for row in results)
                            for sku, frequency in results:
                                if sku and sku.strip():
                                    confidence = round(
                                        0.35 + 0.25 * (frequency / total_matches), 3) if total_matches > 0 else 0.35
                                    suggestions = self._aggregate_sku_suggestions(
                                        suggestions, sku, confidence, f"DB (Fuzzy Series + Original)")
                                    print(
                                        f"    Found in DB (Fuzzy Series + Original): {sku} (Freq: {frequency}, Conf: {confidence})")

                        # STEP 2c: If still no match, try fuzzy series matching with expanded description
                        if not suggestions:
                            print("    No match with original description, trying fuzzy series matching with expanded description...")
                            cursor.execute("""
                                SELECT sku, COUNT(*) as frequency
                                FROM historical_parts
                                WHERE vin_make = ? AND vin_year = ? AND vin_series LIKE ? AND normalized_description = ?
                                GROUP BY sku
                            """, (vin_make, vin_year, f'%{vin_series}%', normalized_expanded))
                            results = cursor.fetchall()
                            total_matches = sum(row[1] for row in results)
                            for sku, frequency in results:
                                if sku and sku.strip():
                                    confidence = round(
                                        0.35 + 0.25 * (frequency / total_matches), 3) if total_matches > 0 else 0.35
                                    suggestions = self._aggregate_sku_suggestions(
                                        suggestions, sku, confidence, f"DB (Fuzzy Series + Expanded)")
                                    print(
                                        f"    Found in DB (Fuzzy Series + Expanded): {sku} (Freq: {frequency}, Conf: {confidence})")

                        # STEP 3: If still no match, try fuzzy description matching with exact series
                        if not suggestions:
                            print("    No exact description match, trying fuzzy description matching...")
                            # Get all descriptions for this make/year/series combination
                            cursor.execute("""
                                SELECT sku, normalized_description, COUNT(*) as frequency
                                FROM historical_parts
                                WHERE vin_make = ? AND vin_year = ? AND vin_series = ?
                                GROUP BY sku, normalized_description
                            """, (vin_make, vin_year, vin_series))
                            all_results = cursor.fetchall()

                            print(f"    🔄 Applying unified preprocessing for Database fuzzy description matching...")

                            # Apply unified preprocessing to input descriptions
                            preprocessed_original = self.unified_text_preprocessing(original_desc)
                            preprocessed_expanded = self.unified_text_preprocessing(expanded_desc)

                            # Group results by SKU to apply consensus logic
                            sku_matches = {}  # sku -> [(similarity, frequency, db_desc, match_type)]

                            # Apply fuzzy matching with unified preprocessing
                            for sku, db_desc, frequency in all_results:
                                if sku and sku.strip() and db_desc:
                                    # Apply unified preprocessing to database description
                                    preprocessed_db_desc = self.unified_text_preprocessing(db_desc)

                                    # Calculate similarity with preprocessed descriptions
                                    similarity_orig = self._calculate_description_similarity(preprocessed_original, preprocessed_db_desc)
                                    similarity_exp = self._calculate_description_similarity(preprocessed_expanded, preprocessed_db_desc)

                                    # Use the better similarity score
                                    if similarity_orig >= similarity_exp and similarity_orig >= 0.7:
                                        if sku not in sku_matches:
                                            sku_matches[sku] = []
                                        sku_matches[sku].append((similarity_orig, frequency, preprocessed_db_desc, "original"))
                                    elif similarity_exp >= 0.7:
                                        if sku not in sku_matches:
                                            sku_matches[sku] = []
                                        sku_matches[sku].append((similarity_exp, frequency, preprocessed_db_desc, "expanded"))

                            # Process SKU matches with frequency-based confidence and consensus logic
                            for sku, matches in sku_matches.items():
                                # Sum frequencies for this SKU across all matching descriptions
                                total_frequency = sum(match[1] for match in matches)
                                best_match = max(matches, key=lambda x: x[0])  # Best similarity
                                similarity, frequency, db_desc, match_type = best_match

                                # Use frequency-based confidence instead of similarity-based
                                confidence = self.calculate_frequency_based_confidence(total_frequency, "DB (Unified Fuzzy)")

                                # Adjust confidence based on similarity quality
                                similarity_bonus = (similarity - 0.7) * 0.1  # Up to 0.03 bonus for perfect similarity
                                final_confidence = round(confidence + similarity_bonus, 3)

                                suggestions = self._aggregate_sku_suggestions(
                                    suggestions, sku, final_confidence, f"DB (Unified Fuzzy {match_type.title()}: {similarity:.2f})")
                                print(
                                    f"    ✅ Found in DB (Unified Fuzzy {match_type.title()}): {sku} (Sim: {similarity:.2f}, Total Freq: {total_frequency}, Conf: {final_confidence})")
                                print(f"      Input: '{preprocessed_original if match_type == 'original' else preprocessed_expanded}' vs DB: '{db_desc}'")

                    except Exception as db_err:
                        print(
                            f"    Error querying SQLite DB (4-param): {db_err}")

                # Fallback: 3-parameter search if no results (Make, Year, Series only)
                if not suggestions and vin_year is not None and vin_series != 'N/A':
                    print(
                        f"  Fallback SQLite search (3-param: Make, Year, Series)...")
                    try:
                        # STEP 1: Try exact series match with fuzzy description
                        cursor.execute("""
                            SELECT sku, normalized_description, COUNT(*) as frequency
                            FROM historical_parts
                            WHERE vin_make = ? AND vin_year = ? AND vin_series = ?
                            GROUP BY sku, normalized_description
                            ORDER BY frequency DESC
                        """, (vin_make, vin_year, vin_series))
                        results = cursor.fetchall()

                        print(f"    🔄 Applying unified preprocessing for 3-param fuzzy matching...")

                        # Apply unified preprocessing to input descriptions
                        preprocessed_original = self.unified_text_preprocessing(original_desc)
                        preprocessed_expanded = self.unified_text_preprocessing(expanded_desc)

                        # Group results by SKU to apply consensus logic for 3-param search
                        sku_matches = {}  # sku -> [(similarity, frequency, db_desc, match_type)]

                        # Apply fuzzy description matching with unified preprocessing
                        for sku, db_desc, frequency in results:
                            if sku and sku.strip() and db_desc:
                                # Apply unified preprocessing to database description
                                preprocessed_db_desc = self.unified_text_preprocessing(db_desc)

                                similarity_orig = self._calculate_description_similarity(preprocessed_original, preprocessed_db_desc)
                                similarity_exp = self._calculate_description_similarity(preprocessed_expanded, preprocessed_db_desc)

                                # Use the better similarity score (lower threshold for 3-param fallback)
                                if similarity_orig >= similarity_exp and similarity_orig >= 0.6:
                                    if sku not in sku_matches:
                                        sku_matches[sku] = []
                                    sku_matches[sku].append((similarity_orig, frequency, preprocessed_db_desc, "original"))
                                elif similarity_exp >= 0.6:
                                    if sku not in sku_matches:
                                        sku_matches[sku] = []
                                    sku_matches[sku].append((similarity_exp, frequency, preprocessed_db_desc, "expanded"))

                        # Apply consensus logic to 3-param results (more lenient for fallback)
                        if sku_matches:
                            # Convert to frequency pairs for consensus analysis
                            sku_frequency_pairs = []
                            for sku, matches in sku_matches.items():
                                total_frequency = sum(match[1] for match in matches)
                                sku_frequency_pairs.append((sku, total_frequency))

                            # Apply consensus logic with lower threshold for fallback
                            consensus_skus = self.apply_consensus_logic(sku_frequency_pairs, min_consensus_ratio=0.4)

                            # Process consensus SKUs
                            for sku, total_frequency in consensus_skus:
                                if sku in sku_matches:
                                    best_match = max(sku_matches[sku], key=lambda x: x[0])  # Best similarity
                                    similarity, frequency, db_desc, match_type = best_match

                                    # Use frequency-based confidence for 3-param fallback
                                    confidence = self.calculate_frequency_based_confidence(total_frequency, "DB (3-param)")

                                    # Small similarity bonus for 3-param fallback
                                    similarity_bonus = (similarity - 0.6) * 0.05  # Up to 0.02 bonus
                                    final_confidence = round(confidence + similarity_bonus, 3)

                                    suggestions = self._aggregate_sku_suggestions(
                                        suggestions, sku, final_confidence, f"DB (3-param Unified {match_type.title()}: {similarity:.2f})")
                                    print(
                                        f"    ✅ Found in DB (3-param Unified {match_type.title()}): {sku} (Sim: {similarity:.2f}, Total Freq: {total_frequency}, Conf: {final_confidence})")

                        # STEP 2: If still no results, try fuzzy series + fuzzy description
                        if not suggestions:
                            print(
                                f"  Final Fallback SQLite search (Fuzzy Series: Make, Year, Series LIKE '%{vin_series}%')...")
                            cursor.execute("""
                                SELECT sku, normalized_description, COUNT(*) as frequency
                                FROM historical_parts
                                WHERE vin_make = ? AND vin_year = ? AND vin_series LIKE ?
                                GROUP BY sku, normalized_description
                                ORDER BY frequency DESC
                            """, (vin_make, vin_year, f'%{vin_series}%'))
                            fuzzy_results = cursor.fetchall()

                            print(f"    🔄 Applying unified preprocessing for fuzzy series+description matching...")

                            # Apply unified preprocessing to input descriptions
                            preprocessed_original = self.unified_text_preprocessing(original_desc)
                            preprocessed_expanded = self.unified_text_preprocessing(expanded_desc)

                            # Apply fuzzy description matching with unified preprocessing and lower threshold
                            for sku, db_desc, frequency in fuzzy_results:
                                if sku and sku.strip() and db_desc:
                                    # Apply unified preprocessing to database description
                                    preprocessed_db_desc = self.unified_text_preprocessing(db_desc)

                                    similarity_orig = self._calculate_description_similarity(preprocessed_original, preprocessed_db_desc)
                                    similarity_exp = self._calculate_description_similarity(preprocessed_expanded, preprocessed_db_desc)

                                    # Use the better similarity score with higher confidence for unified preprocessing
                                    if similarity_orig >= similarity_exp and similarity_orig >= 0.5:
                                        confidence = round(0.2 + 0.3 * similarity_orig, 3)  # Higher confidence for unified preprocessing
                                        suggestions = self._aggregate_sku_suggestions(
                                            suggestions, sku, confidence, f"DB (Unified Fuzzy Series+Desc Orig: {similarity_orig:.2f})")
                                        print(
                                            f"    Found in DB (Unified Fuzzy Series+Desc Orig): {sku} (Sim: {similarity_orig:.2f}, Freq: {frequency}, Conf: {confidence})")
                                    elif similarity_exp >= 0.5:
                                        confidence = round(0.15 + 0.25 * similarity_exp, 3)  # Higher confidence for unified preprocessing
                                        suggestions = self._aggregate_sku_suggestions(
                                            suggestions, sku, confidence, f"DB (Unified Fuzzy Series+Desc Exp: {similarity_exp:.2f})")
                                        print(
                                            f"    Found in DB (Unified Fuzzy Series+Desc Exp): {sku} (Sim: {similarity_exp:.2f}, Freq: {frequency}, Conf: {confidence})")

                    except Exception as db_err:
                        print(
                            f"    Error querying SQLite DB (3-param): {db_err}")







                sorted_suggestions = sorted(
                    suggestions.items(), key=lambda item: item[1]['confidence'], reverse=True)
                self.current_suggestions[original_desc] = sorted_suggestions
                print(
                    f"  Suggestions for '{original_desc}': {sorted_suggestions}")

        except Exception as e:
            messagebox.showerror(
                "Search Error", f"An error occurred during search: {e}")
            print(f"Error during search phase: {e}")
            ttk.Label(self.scrollable_frame,
                      text=f"Error during search: {e}").pack(anchor="nw")
        finally:
            if db_conn:
                db_conn.close()
                print("Database connection closed.")

        # --- Update Results Display ---
        self.display_results()  # Call method to update GUI

    def display_results(self):
        """Updates the vehicle details frame and scrollable results frame with SKU suggestions."""
        # Clear previous results
        self._clear_results_area()

        # Update Vehicle Details in the right column frame
        # First, clear any existing content in the vehicle details frame
        for widget in self.vehicle_details_frame.winfo_children():
            widget.destroy()

        # Display Vehicle Details in the right column frame
        if self.vehicle_details:
            vin = self.vin_entry.get().strip().upper()
            ttk.Label(self.vehicle_details_frame, text=f"VIN: {vin}").pack(
                anchor="w", padx=5, pady=2)
            ttk.Label(
                self.vehicle_details_frame, text=f"Predicted Make: {self.vehicle_details.get('Make', 'N/A')}").pack(anchor="w", padx=5, pady=2)
            ttk.Label(
                self.vehicle_details_frame, text=f"Predicted Year: {self.vehicle_details.get('Model Year', 'N/A')}").pack(anchor="w", padx=5, pady=2)
            ttk.Label(
                self.vehicle_details_frame, text=f"Predicted Series: {self.vehicle_details.get('Series', 'N/A')}").pack(anchor="w", padx=5, pady=2)
        else:
            ttk.Label(self.vehicle_details_frame,
                      text="Vehicle details could not be predicted.").pack(anchor="w", padx=5, pady=5)

        # Display SKU Suggestions (Tasks 5.2, 5.3, 5.4) - Remains the same logic
        if not self.processed_parts:
            ttk.Label(self.scrollable_frame,
                      text="\nNo part descriptions were entered.").pack(anchor="nw")
        elif not self.current_suggestions and self.processed_parts:
            ttk.Label(self.scrollable_frame,
                      text="\nNo suggestions found for the entered descriptions.").pack(anchor="nw")
        elif self.current_suggestions:
            # Add a small separator but no redundant header
            ttk.Separator(self.scrollable_frame, orient='horizontal').pack(
                fill='x', pady=5)

            # Create a container frame for the responsive grid
            self.results_grid_container = ttk.Frame(self.scrollable_frame)
            self.results_grid_container.pack(
                fill="both", expand=True, padx=5, pady=5)  # Fill both to get width

            # Set a minimum width to ensure proper layout calculation
            self.results_grid_container.config(
                width=self.root.winfo_width() - 50)

            # Clear previous part frames widgets list
            self.part_frames_widgets = []

            # Reset the current number of columns to force recalculation
            self.current_num_columns = 0

            # Create frames for each part description
            for part_info in self.processed_parts:
                original_desc = part_info["original"]
                normalized_original = part_info["normalized_original"]

                # Use the normalized (expanded) version for display, but capitalize it properly
                display_desc = normalized_original.upper() if normalized_original else original_desc

                # Get suggestions and filter out any empty SKUs that might have slipped through
                suggestions_list = [(sku, info) for sku, info in self.current_suggestions.get(original_desc, [])
                                    if sku and sku.strip()][:5]

                part_frame = ttk.LabelFrame(
                    self.results_grid_container, text=f"{display_desc}", padding=5)
                self.part_frames_widgets.append(part_frame)
                part_frame.columnconfigure(0, weight=1)
                part_frame.config(width=200)

                if suggestions_list:
                    # Check for auto-preselection (1.00 confidence)
                    auto_preselect_sku = None
                    top_suggestion = suggestions_list[0] if suggestions_list else None
                    if top_suggestion:
                        top_sku, top_info = top_suggestion
                        top_confidence = top_info.get('confidence', 0)
                        if top_confidence >= 1.00:
                            auto_preselect_sku = top_sku
                            print(f"🎯 Auto-preselecting {top_sku} for '{original_desc}' (Confidence: {top_confidence:.2f})")

                    self.selection_vars[original_desc] = tk.StringVar(
                        value=auto_preselect_sku)  # Auto-preselect if 1.00 confidence
                    self.manual_sku_vars = getattr(self, 'manual_sku_vars', {})
                    self.manual_sku_vars[original_desc] = tk.StringVar()

                    # Add preselection indicator if auto-preselected
                    if auto_preselect_sku:
                        preselect_label = ttk.Label(
                            part_frame,
                            text="🎯 Auto-selected (1.00 confidence) - You can change this selection:",
                            foreground="green",
                            font=("", 9, "italic")
                        )
                        preselect_label.pack(anchor="w", padx=5, pady=2)

                    ttk.Separator(part_frame, orient='horizontal').pack(
                        fill='x', pady=5)

                    # Add radio buttons for each suggestion
                    for sku, info in suggestions_list:
                        conf = info.get('confidence', 0)
                        source = info.get('source', '')
                        all_sources = info.get('all_sources', source)

                        # Show all sources if multiple, otherwise just the main source
                        display_source = all_sources if all_sources != source else source

                        rb = ttk.Radiobutton(
                            part_frame,
                            text=f"{sku} ({self._format_confidence_percentage(conf)}, {display_source})",
                            variable=self.selection_vars[original_desc],
                            value=sku
                        )
                        rb.pack(anchor="w", padx=5, pady=2)

                    # Manual entry option
                    rb_none_frame = ttk.Frame(part_frame)
                    rb_none_frame.pack(anchor="w", padx=5, pady=2, fill="x")
                    rb_none = ttk.Radiobutton(
                        rb_none_frame,
                        text="Manual Entry:",
                        variable=self.selection_vars[original_desc],
                        value="MANUAL"
                    )
                    rb_none.pack(side=tk.LEFT, padx=(0, 5))
                    manual_entry = ttk.Entry(
                        rb_none_frame,
                        textvariable=self.manual_sku_vars[original_desc],
                        width=15
                    )
                    manual_entry.pack(side=tk.LEFT, padx=5)

                    def _on_manual_entry_change(*_, desc=original_desc):
                        if self.manual_sku_vars[desc].get().strip():
                            self.selection_vars[desc].set("MANUAL")
                    self.manual_sku_vars[original_desc].trace_add(
                        "write", _on_manual_entry_change)

                    # 'I don't know' option
                    rb_unknown_frame = ttk.Frame(part_frame)
                    rb_unknown_frame.pack(anchor="w", padx=5, pady=2, fill="x")
                    rb_unknown = ttk.Radiobutton(
                        rb_unknown_frame,
                        text="I don't know the SKU",
                        variable=self.selection_vars[original_desc],
                        value="UNKNOWN"
                    )
                    rb_unknown.pack(side=tk.LEFT, fill="x", expand=True)
                else:
                    # No suggestions, only manual and unknown options
                    self.selection_vars[original_desc] = tk.StringVar(
                        value=None)
                    self.manual_sku_vars = getattr(self, 'manual_sku_vars', {})
                    self.manual_sku_vars[original_desc] = tk.StringVar()
                    rb_none_frame = ttk.Frame(part_frame)
                    rb_none_frame.pack(anchor="w", padx=5, pady=2, fill="x")
                    rb_none = ttk.Radiobutton(
                        rb_none_frame,
                        text="Manual Entry:",
                        variable=self.selection_vars[original_desc],
                        value="MANUAL"
                    )
                    rb_none.pack(side=tk.LEFT, padx=(0, 5))
                    manual_entry = ttk.Entry(
                        rb_none_frame,
                        textvariable=self.manual_sku_vars[original_desc],
                        width=15
                    )
                    manual_entry.pack(side=tk.LEFT, padx=5)

                    def _on_manual_entry_change(*_, desc=original_desc):
                        if self.manual_sku_vars[desc].get().strip():
                            self.selection_vars[desc].set("MANUAL")
                    self.manual_sku_vars[original_desc].trace_add(
                        "write", _on_manual_entry_change)
                    rb_unknown_frame = ttk.Frame(part_frame)
                    rb_unknown_frame.pack(anchor="w", padx=5, pady=2, fill="x")
                    rb_unknown = ttk.Radiobutton(
                        rb_unknown_frame,
                        text="I don't know the SKU",
                        variable=self.selection_vars[original_desc],
                        value="UNKNOWN"
                    )
                    rb_unknown.pack(side=tk.LEFT, fill="x", expand=True)

            # Initial layout + bind configure event for responsiveness
            print("Initializing responsive layout...")
            self.root.update_idletasks()  # Force geometry update before calculating layout
            self._resize_results_columns()  # Perform initial layout
            self.results_grid_container.bind(
                "<Configure>", self._on_results_configure)
            self.root.bind("<Configure>", self._on_results_configure)
            if self.processed_parts:
                self.save_button.config(state=tk.NORMAL)
            else:
                self.save_button.config(state=tk.DISABLED)

    def _on_results_configure(self, _=None):
        # This method is called when the results_grid_container is resized
        # We add a small delay (debounce) to avoid excessive re-layouts during rapid resizing
        # The event parameter is unused but required by the bind method
        if hasattr(self, '_after_id_resize'):
            self.root.after_cancel(self._after_id_resize)
        self._after_id_resize = self.root.after(
            100, self._resize_results_columns)

    def _resize_results_columns(self):
        """Recalculates and applies the grid layout for part_frames to create a responsive layout."""
        if not hasattr(self, 'results_grid_container') or not self.results_grid_container.winfo_exists():
            return
        if not self.part_frames_widgets:  # No items to grid
            return

        # Get the current width of the container
        container_width = self.results_grid_container.winfo_width()

        # If the container width is not yet initialized, use the root window width as a fallback
        if container_width <= 1:
            container_width = self.root.winfo_width() - 40  # Subtract some padding

        # Get the actual width of a part frame by measuring the first one
        # This ensures we use the real width rather than an estimate
        if self.part_frames_widgets:
            # Update the first widget to ensure its size is calculated
            self.part_frames_widgets[0].update_idletasks()
            # Add padding
            actual_item_width = self.part_frames_widgets[0].winfo_reqwidth(
            ) + 10
        else:
            actual_item_width = 220  # Fallback if no widgets exist

        # Calculate the number of columns that can fit completely
        # We only want to show complete columns (no partial columns)
        if container_width <= actual_item_width:
            num_columns = 1
        else:
            # Calculate how many complete columns can fit
            num_columns = max(1, int(container_width / actual_item_width))

            # Ensure we don't create more columns than we have items
            num_columns = min(num_columns, len(self.part_frames_widgets))

        print(
            f"Container width: {container_width}, Item width: {actual_item_width}, Complete columns: {num_columns}")

        # Check if we need to update the layout
        layout_needs_update = (
            num_columns != self.current_num_columns or
            self.current_num_columns == 0 or
            len(self.results_grid_container.grid_slaves()) != len(
                self.part_frames_widgets)
        )

        if not layout_needs_update:
            # Further check if all columns have the correct weight
            all_weights_correct = True
            for i in range(num_columns):
                if self.results_grid_container.grid_columnconfigure(i).get('weight', '0') != '1':
                    all_weights_correct = False
                    break

            if all_weights_correct:
                return  # No layout change needed

        # Update the layout

        # 1. Clear all existing column configurations
        current_configured_cols = max(self.current_num_columns,
                                      self.results_grid_container.grid_size()[0])
        for i in range(current_configured_cols):
            self.results_grid_container.columnconfigure(i, weight=0)

        # 2. Remove all widgets from the grid
        for widget in self.results_grid_container.grid_slaves():
            widget.grid_forget()

        # 3. Configure the new columns with equal weight
        for i in range(num_columns):
            self.results_grid_container.columnconfigure(i, weight=1)

        # 4. Update the current number of columns
        self.current_num_columns = num_columns

        # 5. Re-grid all the part frames in the new layout
        for idx, frame_widget in enumerate(self.part_frames_widgets):
            row = idx // num_columns
            col = idx % num_columns
            frame_widget.grid(row=row, column=col, padx=5,
                              pady=5, sticky="nsew")

        print(
            f"Updated layout to {num_columns} complete columns with {len(self.part_frames_widgets)} items")

    # The _toggle_manual_entry method is no longer needed as manual entry is always visible

    # The _confirm_manual_sku method is no longer needed as the manual entry
    # automatically selects the radio button and will be saved with the "Save Confirmed Selections" button

    def save_selections_handler(self):
        """
        Handles the 'Save Confirmed Selections' button click.
        Gathers selected SKUs, adds them to the in-memory Maestro data,
        and writes the updated data back to Maestro.xlsx.
        (Corresponds to Tasks 5.6, 5.7, 5.8)
        """
        global maestro_data_global
        print("\n--- 'Save Confirmed Selections' clicked ---")
        if not self.vehicle_details or not self.processed_parts:
            messagebox.showwarning(
                "Cannot Save", "No valid search results available to save.")
            return

        selections_to_save = []
        for part_info in self.processed_parts:
            original_desc = part_info["original"]
            selected_sku_var = self.selection_vars.get(original_desc)
            if selected_sku_var:
                selected_sku = selected_sku_var.get()
                if selected_sku:
                    # Check if this is a manually entered SKU or unknown
                    if selected_sku == "UNKNOWN":
                        print(f"Skipping UNKNOWN SKU for '{original_desc}' - not saving to Maestro")
                        continue  # Skip UNKNOWN entries - don't save them to Maestro
                    elif selected_sku == "MANUAL":
                        # Get the actual SKU from the manual entry field
                        manual_sku_var = self.manual_sku_vars.get(original_desc)
                        if manual_sku_var:
                            actual_manual_sku = manual_sku_var.get().strip()
                            if actual_manual_sku:
                                # Validate the manually entered SKU
                                if not self._is_valid_sku(actual_manual_sku):
                                    print(f"Skipping invalid manual SKU '{actual_manual_sku}' for '{original_desc}' - not saving to Maestro")
                                    continue
                                selected_sku = actual_manual_sku  # Use the actual SKU, not "MANUAL"
                                source = "UserManualEntry"
                                print(f"Manual SKU entered for '{original_desc}': {selected_sku}")
                            else:
                                print(f"Warning: Manual entry selected but no SKU provided for '{original_desc}'")
                                continue  # Skip this entry if no manual SKU provided
                        else:
                            print(f"Warning: Manual entry selected but no manual_sku_var found for '{original_desc}'")
                            continue  # Skip this entry
                    else:
                        # This is a suggested SKU that was selected
                        # Validate the selected SKU before saving
                        if not self._is_valid_sku(selected_sku):
                            print(f"Skipping invalid suggested SKU '{selected_sku}' for '{original_desc}' - not saving to Maestro")
                            continue

                        is_manual = True
                        for sku, _ in self.current_suggestions.get(original_desc, []):
                            if sku == selected_sku:
                                is_manual = False
                                break
                        source = "UserManualEntry" if is_manual else "UserConfirmed"

                    print(f"Selected for '{original_desc}': SKU = {selected_sku} (Source: {source})")

                    part_data = next(
                        (p for p in self.processed_parts if p["original"] == original_desc), None)
                    if part_data:
                        selections_to_save.append({
                            "vin_details": self.vehicle_details,
                            "original_description": original_desc,
                            "normalized_description": part_data["normalized_expanded"],  # Use expanded for consistency
                            "equivalencia_id": part_data["equivalencia_id"],
                            "confirmed_sku": selected_sku,
                            "source": source
                        })
                else:
                    print(f"Selected for '{original_desc}': None")
            else:
                print(f"No selection variable found for '{original_desc}'")

        if not selections_to_save:
            messagebox.showinfo(
                "Nothing to Save", "No valid SKUs were selected for confirmation.\n\nNote: UNKNOWN selections and invalid SKUs are not saved to preserve data quality.")
            return

        added_count = 0
        skipped_count = 0
        max_id = 0
        if maestro_data_global:
            ids = [entry.get('Maestro_ID', 0) for entry in maestro_data_global if isinstance(
                entry.get('Maestro_ID'), int)]
            if ids:
                max_id = max(ids)
        next_id = max_id + 1

        for selection in selections_to_save:
            is_duplicate = False
            for existing_entry in maestro_data_global:
                # Check for duplicates based on 4-parameter approach (no VIN_Model, VIN_BodyStyle, Equivalencia_Row_ID)
                if (existing_entry.get('VIN_Make') == selection['vin_details'].get('Make') and
                    existing_entry.get('VIN_Year_Min') == selection['vin_details'].get('Model Year') and
                    existing_entry.get('VIN_Series_Trim') == selection['vin_details'].get('Series') and
                    existing_entry.get('Normalized_Description_Input') == selection['normalized_description'] and
                        existing_entry.get('Confirmed_SKU') == selection['confirmed_sku']):
                    is_duplicate = True
                    break
            if not is_duplicate:
                # Extract and convert year to integer
                model_year = selection['vin_details'].get('Model Year')
                if isinstance(model_year, (list, tuple, np.ndarray)):
                    model_year = model_year[0] if len(model_year) > 0 else None
                if isinstance(model_year, str):
                    try:
                        model_year = int(model_year)
                    except (ValueError, TypeError):
                        model_year = None

                new_entry = {
                    'Maestro_ID': next_id,
                    'VIN_Make': selection['vin_details'].get('Make'),
                    'VIN_Year_Min': model_year,
                    'VIN_Year_Max': model_year,
                    'VIN_Series_Trim': selection['vin_details'].get('Series'),
                    'Original_Description_Input': selection['original_description'],
                    'Normalized_Description_Input': selection['normalized_description'],
                    'Confirmed_SKU': selection['confirmed_sku'],
                    'Confidence': 1.0,
                    'Source': selection.get('source', 'UserConfirmed'),
                    'Date_Added': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                maestro_data_global.append(new_entry)
                added_count += 1
                next_id += 1
            else:
                skipped_count += 1
                print(
                    f"Skipped duplicate: {selection['original_description']} - {selection['confirmed_sku']}")

        if added_count > 0:
            print(
                f"Attempting to save {added_count} new entries to {DEFAULT_MAESTRO_PATH}...")
            try:
                # Removed columns: VIN_Model, VIN_BodyStyle, Equivalencia_Row_ID
                maestro_columns = [
                    'Maestro_ID', 'VIN_Make', 'VIN_Year_Min', 'VIN_Year_Max',
                    'VIN_Series_Trim', 'Original_Description_Input',
                    'Normalized_Description_Input', 'Confirmed_SKU',
                    'Confidence', 'Source', 'Date_Added'
                ]
                df_to_save = pd.DataFrame(
                    maestro_data_global, columns=maestro_columns)
                df_to_save.to_excel(DEFAULT_MAESTRO_PATH, index=False)
                messagebox.showinfo(
                    "Save Successful", f"{added_count} new confirmation(s) saved to Maestro.xlsx.")
                print(
                    f"Successfully saved {added_count} new entries. Total Maestro entries: {len(maestro_data_global)}")
            except Exception as e:
                messagebox.showerror(
                    "Save Error", f"Failed to write to Maestro.xlsx: {e}")
                print(f"Error writing Maestro.xlsx: {e}")
        elif skipped_count > 0:
            messagebox.showinfo(
                "Already Saved", "The selected confirmations were already present in Maestro.xlsx.")
            print(f"Skipped {skipped_count} duplicate entries.")
        else:
            messagebox.showinfo(
                "Nothing Saved", "No new confirmations were added.")

    # Removed decode_vin_nhtsa


if __name__ == '__main__':
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print(
        f"Expected Equivalencias.xlsx at: {os.path.join(current_dir, DEFAULT_EQUIVALENCIAS_PATH)}")
    print(
        f"Expected/Creating Maestro.xlsx at: {os.path.join(current_dir, DEFAULT_MAESTRO_PATH)}")
    print(
        f"Expected fixacar_history.db at: {os.path.join(current_dir, DEFAULT_DB_PATH)}")
    # Removed the print statement that caused the error

    root = tk.Tk()
    root.title("Fixacar SKU Finder v2.0 (with VIN Predictor)")
    root.geometry("1200x800")  # Set a reasonable default size

    # Maximize the window on startup to ensure all buttons are visible
    root.state('zoomed')  # Windows equivalent of maximized

    app = FixacarApp(root)
    root.mainloop()
