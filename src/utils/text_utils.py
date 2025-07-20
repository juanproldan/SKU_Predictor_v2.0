import unicodedata
import re
from typing import Optional, List, Tuple

# Import our fuzzy matching utilities
try:
    from utils.fuzzy_matcher import fuzzy_normalize_text, get_fuzzy_matches, find_best_match
except ImportError:
    # Fallback for direct execution if src is not in path
    try:
        from .fuzzy_matcher import fuzzy_normalize_text, get_fuzzy_matches, find_best_match
    except ImportError:
        # If fuzzy_matcher is not available, define placeholder functions
        def fuzzy_normalize_text(text: str) -> str:
            """Placeholder for fuzzy_normalize_text if module not available"""
            return normalize_text(text)

        def get_fuzzy_matches(query: str, candidates: List[str], threshold: float = 0.7) -> List[Tuple[str, float]]:
            """Placeholder for get_fuzzy_matches if module not available"""
            return [(c, 1.0) for c in candidates if c == query]

        def find_best_match(query: str, candidates: List[str], threshold: float = 0.7) -> Optional[Tuple[str, float]]:
            """Placeholder for find_best_match if module not available"""
            if query in candidates:
                return (query, 1.0)
            return None


def normalize_text(text: str, use_fuzzy: bool = False, expand_linguistic_variations: bool = True) -> str:
    """
    Normalizes a text string according to the project's requirements:
    - Converts to lowercase.
    - Removes leading/trailing whitespace.
    - Standardizes internal whitespace (multiple spaces to one).
    - Removes common punctuation (keeps alphanumeric and spaces).
    - Handles accented characters/diacritics (e.g., converts 'á' to 'a').
    - Optionally expands abbreviations, gender, and plural variations.

    Args:
        text: The input text to normalize
        use_fuzzy: Whether to use enhanced fuzzy normalization (default: False)
        expand_linguistic_variations: Whether to expand abbreviations, gender, plurals (default: True)

    Returns:
        Normalized text string
    """
    if not isinstance(text, str):
        return ""  # Or raise an error, depending on desired handling for non-strings

    if use_fuzzy:
        # Use the enhanced fuzzy normalization
        return fuzzy_normalize_text(text)

    # Standard normalization (original implementation)
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove leading/trailing whitespace
    text = text.strip()

    # 3. Normalize accented characters
    # Decompose into base character and combining diacritical marks, then remove marks
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # 4. Remove common punctuation (keeps alphanumeric characters and spaces)
    # This regex will remove anything that's not a letter, number, or whitespace.
    # If specific punctuation needs to be kept or replaced differently, adjust the regex.
    text = re.sub(r'[^\w\s]', '', text)  # \w is alphanumeric + underscore

    # 5. Standardize internal whitespace (multiple spaces/tabs/newlines to a single space)
    # strip again in case regex leaves leading/trailing space
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Expand linguistic variations (abbreviations, gender, plurals) if requested
    if expand_linguistic_variations:
        text = expand_linguistic_variations_text(text)

    return text


def get_noun_gender(word: str) -> str:
    """
    Determines the gender of Spanish automotive nouns.

    Args:
        word: The noun to analyze

    Returns:
        'masculine' or 'feminine'
    """
    word = word.lower()

    # Explicit masculine automotive parts
    masculine_parts = {
        'guardafango', 'guardabarros', 'paragolpes', 'parachoques', 'espejo',
        'retrovisor', 'faro', 'piloto', 'intermitente', 'absorbedor',
        'amortiguador', 'radiador', 'electroventilador', 'ventilador',
        'filtro', 'motor', 'alternador', 'compresor', 'condensador',
        'evaporador', 'intercooler', 'turbo', 'catalizador', 'silenciador',
        'tubo', 'colector', 'multiple', 'deposito', 'tanque', 'carter',
        'diferencial', 'eje', 'semieje', 'cardan', 'tensor', 'rodamiento',
        'cojinete', 'reten', 'soporte', 'brazo', 'triangulo', 'rotula',
        'estabilizador', 'bieleta', 'tirante', 'travesano', 'larguero',
        'refuerzo', 'marco', 'chasis', 'bastidor', 'capo', 'maletero',
        'techo', 'techo', 'cristal', 'vidrio', 'elevalunas', 'regulador'
    }

    # Explicit feminine automotive parts
    feminine_parts = {
        'farola', 'luz', 'lampara', 'bombilla', 'optica', 'tulipa',
        'puerta', 'ventana', 'ventanilla', 'luna', 'aleta', 'chapa',
        'carroceria', 'pintura', 'moldura', 'maneta', 'manilla', 'cerradura',
        'bisagra', 'goma', 'junta', 'correa', 'cadena', 'polea', 'rueda',
        'llanta', 'cubierta', 'neumatico', 'camara', 'valvula', 'bujia',
        'bobina', 'bateria', 'dinamo', 'bomba', 'manguera', 'tuberia',
        'conexion', 'union', 'abrazadera', 'brida', 'tapa', 'cubierta',
        'carcasa', 'caja', 'palanca', 'varilla', 'barra', 'biela',
        'cruceta', 'horquilla', 'zapata', 'pastilla', 'lona', 'banda',
        'placa', 'chapa', 'plancha', 'rejilla', 'parrilla', 'mascara',
        'guia'  # Added: GUIA is feminine (la guía)
    }

    if word in masculine_parts:
        return 'masculine'
    elif word in feminine_parts:
        return 'feminine'
    else:
        # Default rules based on Spanish grammar
        if word.endswith(('o', 'or', 'aje', 'an', 'en', 'in', 'on', 'un')):
            return 'masculine'
        elif word.endswith(('a', 'ion', 'dad', 'tad', 'tud', 'ez', 'eza', 'sis', 'itis')):
            return 'feminine'
        else:
            # Default to masculine for unknown cases
            return 'masculine'


def expand_linguistic_variations_text(text: str) -> str:
    """
    Expands linguistic variations in Spanish automotive text:
    - Context-aware abbreviations: handles 'd' and 't' based on part type
    - Gender-aware abbreviations: izq -> izquierdo/izquierda based on noun gender
    - Standard abbreviations with proper gender agreement
    - Plural variations: farolas -> farola (normalize to singular, except specific cases)

    This is independent of industry synonyms and handles basic Spanish linguistic patterns.
    """
    if not text:
        return text

    words = text.split()
    expanded_words = []

    # First pass: Handle special abbreviation patterns (like "D I" = "DELANTERO IZQUIERDO")
    words = handle_abbreviation_patterns(words)

    for i, word in enumerate(words):
        # Handle context-dependent single-letter abbreviations
        if word in ['d', 't'] and i > 0:
            expanded_word = expand_context_dependent_abbreviation(word, words[i-1])
        # Handle gender-dependent abbreviations (left/right and front/rear)
        elif word in ['i', 'iz', 'izq', 'der', 'dere', 'derec', 'derech', 'del', 'delan', 'delant', 'delante', 'tra', 'tras', 'trase', 'traser'] and i > 0:
            # Pass complete context for better gender agreement analysis
            expanded_word = expand_gender_dependent_abbreviation(
                word, words[i-1], all_words=words, word_index=i)
        else:
            expanded_word = expand_single_word_linguistic(word)
        expanded_words.append(expanded_word)

    return ' '.join(expanded_words)


def handle_abbreviation_patterns(words: list) -> list:
    """
    Handles special abbreviation patterns that require contextual analysis.

    LOGICAL RULE: A part cannot be both LEFT and RIGHT simultaneously!

    Recognizes ALL abbreviation combinations:
    - Position abbreviations: D, DE, DEL, DELAN, DELANT, T, TR, TRA, TRAS, TRASE
    - Side abbreviations: I, IZ, IZQ, IZQU, DER, DERE, DEREC, DERECH

    Examples:
    - "D I" -> "DELANTERO IZQUIERDO" (front left)
    - "D IZQU" -> "DELANTERO IZQUIERDO" (front left)
    - "DEL IZ" -> "DELANTERO IZQUIERDO" (front left)
    - "T DER" -> "TRASERO DERECHO" (rear right)
    - "TRA I" -> "TRASERO IZQUIERDO" (rear left)

    The system applies logical spatial reasoning to avoid impossible combinations.

    Args:
        words: List of words to process

    Returns:
        List of words with patterns expanded using logical spatial rules
    """
    if len(words) < 2:
        return words

    # Define abbreviation groups
    position_front_abbrevs = {'d', 'de', 'del', 'delan', 'delant', 'delante'}
    position_rear_abbrevs = {'t', 'tr', 'tra', 'tras', 'trase', 'traser'}
    side_left_abbrevs = {'i', 'iz', 'izq', 'izqu', 'izqui', 'izquie', 'izquier'}
    side_right_abbrevs = {'der', 'dere', 'derec', 'derech', 'derecho', 'derecha'}

    result = []
    i = 0

    while i < len(words):
        current_word = words[i].lower()
        next_word = words[i + 1].lower() if i + 1 < len(words) else ""

        # Find main noun for gender agreement
        main_noun = ""
        for j in range(len(words)):
            word = words[j].lower()
            if word in {'guia', 'farola', 'luz', 'puerta', 'aleta', 'chapa', 'rejilla', 'parrilla', 'mascara',
                       'guardafango', 'paragolpes', 'espejo', 'faro', 'absorbedor', 'radiador', 'soporte',
                       'guardapolvo', 'bomper', 'bumper', 'capo', 'tapa', 'cubierta', 'moldura'}:
                main_noun = word
                break

        gender = get_noun_gender(main_noun) if main_noun else 'masculine'

        # Check if current word is a position abbreviation
        is_front = current_word in position_front_abbrevs
        is_rear = current_word in position_rear_abbrevs

        # Check if next word is a side abbreviation
        is_left = next_word in side_left_abbrevs
        is_right = next_word in side_right_abbrevs

        # Handle FRONT + LEFT patterns (D I, DEL IZQU, etc.)
        if is_front and is_left:
            if gender == 'feminine':
                result.append('delantera')
                result.append('izquierda')
            else:
                result.append('delantero')
                result.append('izquierdo')
            i += 2  # Skip both words
        # Handle FRONT + RIGHT patterns (D DER, DEL DERECH, etc.)
        elif is_front and is_right:
            if gender == 'feminine':
                result.append('delantera')
                result.append('derecha')
            else:
                result.append('delantero')
                result.append('derecho')
            i += 2
        # Handle REAR + LEFT patterns (T I, TRA IZQU, etc.)
        elif is_rear and is_left:
            if gender == 'feminine':
                result.append('trasera')
                result.append('izquierda')
            else:
                result.append('trasero')
                result.append('izquierdo')
            i += 2
        # Handle REAR + RIGHT patterns (T DER, TRAS DERECH, etc.)
        elif is_rear and is_right:
            if gender == 'feminine':
                result.append('trasera')
                result.append('derecha')
            else:
                result.append('trasero')
                result.append('derecho')
            i += 2
        # Special case: Handle "D D" as FRONT + RIGHT (since D can be both)
        elif current_word == 'd' and next_word == 'd':
            if gender == 'feminine':
                result.append('delantera')
                result.append('derecha')
            else:
                result.append('delantero')
                result.append('derecho')
            i += 2
        else:
            # No pattern match, keep original word
            result.append(words[i])
            i += 1

    return result


def find_main_noun_in_phrase(words: list, current_index: int) -> str:
    """
    Finds the main noun in a phrase to determine gender agreement.

    In Spanish automotive terms, the main noun is usually the FIRST noun in the phrase.
    All adjectives in the phrase should agree with this main noun.
    Example: "GUIA LATERAL IZQUIERDA PARAGOLPES DELANTERA"
    -> main noun is "GUIA" (feminine), so all adjectives should be feminine

    Args:
        words: List of words in the phrase
        current_index: Index of the current word being processed

    Returns:
        The main noun that should determine gender agreement
    """
    # First, look for the FIRST noun in the entire phrase (main noun)
    for i in range(len(words)):
        word = words[i].lower()
        # Check if this word is a known automotive noun with explicit gender
        if word in {'guia', 'farola', 'luz', 'puerta', 'aleta', 'chapa', 'rejilla', 'parrilla', 'mascara',
                   'guardafango', 'paragolpes', 'espejo', 'faro', 'absorbedor', 'radiador', 'soporte',
                   'bomper', 'bumper', 'capo', 'tapa', 'cubierta', 'moldura', 'maneta', 'cerradura'}:
            return word

    # If no explicit main noun found, look backwards from current position
    for i in range(current_index - 1, -1, -1):
        word = words[i].lower()
        if get_noun_gender(word) in ['masculine', 'feminine']:
            return word

    # Final fallback: use the immediately preceding word
    if current_index > 0:
        return words[current_index - 1].lower()

    return ""


def expand_gender_dependent_abbreviation(abbrev: str, context_word: str, all_words: list = None, word_index: int = -1) -> str:
    """
    Expands gender-dependent abbreviations based on the gender of the main noun in the phrase.

    Args:
        abbrev: The abbreviation ('i', 'iz', 'izq', 'der', 'del', 'tra', etc.)
        context_word: The immediately preceding word (for backward compatibility)
        all_words: Complete list of words in the phrase (for better context analysis)
        word_index: Index of the current abbreviation in all_words

    Returns:
        The expanded form with proper gender agreement
    """
    if not abbrev:
        return abbrev

    # Determine the gender using improved context analysis
    if all_words and word_index >= 0:
        main_noun = find_main_noun_in_phrase(all_words, word_index)
        gender = get_noun_gender(main_noun) if main_noun else 'masculine'
    else:
        # Fallback to old behavior for backward compatibility
        gender = get_noun_gender(context_word) if context_word else 'masculine'

    # Map abbreviations to their masculine and feminine forms
    if abbrev in ['i', 'iz', 'izq']:
        return 'izquierdo' if gender == 'masculine' else 'izquierda'
    elif abbrev in ['der', 'dere', 'derec', 'derech']:
        return 'derecho' if gender == 'masculine' else 'derecha'
    elif abbrev in ['del', 'delan', 'delant', 'delante']:
        return 'delantero' if gender == 'masculine' else 'delantera'
    elif abbrev in ['tra', 'tras', 'trase', 'traser']:
        return 'trasero' if gender == 'masculine' else 'trasera'

    # Return unchanged if not handled
    return abbrev


def expand_context_dependent_abbreviation(abbrev: str, context_word: str) -> str:
    """
    Expands context-dependent single-letter abbreviations based on the automotive part type.

    Args:
        abbrev: The abbreviation ('d' or 't')
        context_word: The preceding word that provides context (e.g., 'farola', 'paragolpes')

    Returns:
        The expanded form based on context
    """
    if not abbrev or not context_word:
        return abbrev

    # Define part categories and their abbreviation meanings
    part_categories = {
        # Parts with LEFT/RIGHT positioning (d = derecha, i = izquierda)
        'lateral_parts': {
            'espejo', 'espejos', 'guardafango', 'guardafangos',
            'guardabarro', 'guardabarros', 'puerta', 'puertas'
        },

        # Parts with FRONT/REAR positioning (d = delantero, t = trasero)
        'longitudinal_parts': {
            'paragolpes', 'paragolpe', 'bomper', 'defensa',
            'traviesa', 'traviesas', 'refuerzo', 'refuerzos'
        },

        # Special case: Lights can have both front/rear AND left/right
        # For compound light descriptions, prioritize front/rear
        'light_parts_frontrear': {
            'luz', 'luces', 'farola', 'farolas', 'faro', 'faros'
        }
    }

    # Determine part category with enhanced context analysis
    part_category = None

    # Check for compound light descriptions (e.g., "luz antiniebla")
    # In these cases, 'd' typically means 'delantera' (front)
    if context_word in part_categories['light_parts_frontrear']:
        # Look at the full context to determine if it's a compound light description
        # This is a simplified approach - in a full implementation, we'd analyze the full text
        part_category = 'longitudinal_parts'  # Treat lights as front/rear by default
    else:
        # Standard category detection
        for category, parts in part_categories.items():
            if context_word in parts:
                part_category = category
                break

    # Expand based on context
    if abbrev == 'd':
        if part_category == 'lateral_parts':
            return 'derecha'  # espejo d = espejo derecha
        elif part_category == 'longitudinal_parts':
            return 'delantero'  # paragolpes d = paragolpes delantero, luz d = luz delantera
        elif part_category == 'light_parts_frontrear':
            return 'delantera'  # luz d = luz delantera (for compound descriptions)
        else:
            # Default to 'derecha' for unknown contexts (most common)
            return 'derecha'

    elif abbrev == 't':
        # 't' almost always means 'trasero' in automotive context
        return 'trasero'

    # Return unchanged if not handled
    return abbrev


def expand_single_word_linguistic(word: str) -> str:
    """
    Expands a single word for linguistic variations.
    Note: This function handles abbreviations and plurals, but NOT gender normalization.
    Gender variations are handled during similarity matching, not normalization.

    For context-dependent abbreviations, this function only handles clear cases.
    Ambiguous single letters are handled by expand_linguistic_variations_text()
    which has access to the full text context.
    """
    if not word:
        return word

    # 1. Abbreviation expansion (directional terms) - CLEAR CASES ONLY
    # Note: Gender-dependent abbreviations (i, iz, izq, der, etc.) are handled
    # by expand_gender_dependent_abbreviation() in the main text processing
    abbreviation_map = {
        # FRONT variations (gender-neutral or context will determine)
        'del': 'delantero',  # Default to masculine, context may override
        'delan': 'delantero',
        'delant': 'delantero',
        'delante': 'delantero',

        # REAR variations (gender-neutral or context will determine)
        'tra': 'trasero',  # Default to masculine, context may override
        'tras': 'trasero',
        'trase': 'trasero',
        'traser': 'trasero',

        # OTHER directional (these are typically gender-neutral)
        'sup': 'superior',
        'super': 'superior',
        'inf': 'inferior',
        'infer': 'inferior',
        'ant': 'anterior',
        'anter': 'anterior',
        'post': 'posterior',
        'poster': 'posterior',
    }

    # Check abbreviations first
    if word in abbreviation_map:
        return abbreviation_map[word]

    # 2. Plural normalization (context-independent)
    plural_map = {
        'farolas': 'farola',
        'luces': 'luz',
        'espejos': 'espejo',
        'puertas': 'puerta',
        'paragolpe': 'paragolpes',  # This one goes to plural (standard form)
        'guardafangos': 'guardafango',
        'guardabarros': 'guardabarro',
    }

    if word in plural_map:
        return plural_map[word]

    # No expansion needed - ambiguous cases handled at text level
    return word


def are_gender_variants(word1: str, word2: str) -> bool:
    """
    Checks if two words are gender variants of each other.
    Returns True if they are masculine/feminine versions of the same word.

    Examples:
    - derecho/derecha -> True
    - izquierdo/izquierda -> True
    - farola/farola -> False (same word)

    CASE-INSENSITIVE: All comparisons are done in lowercase.
    """
    if not word1 or not word2:
        return False

    # Convert to lowercase for case-insensitive comparison
    word1 = word1.lower()
    word2 = word2.lower()

    if word1 == word2:
        return False

    # Define gender pairs (masculine, feminine)
    gender_pairs = [
        ('derecho', 'derecha'),
        ('izquierdo', 'izquierda'),
        ('delantero', 'delantera'),
        ('trasero', 'trasera'),
    ]

    for masculine, feminine in gender_pairs:
        if (word1 == masculine and word2 == feminine) or (word1 == feminine and word2 == masculine):
            return True

    return False


if __name__ == '__main__':
    # Test cases
    test_strings = {
        "  Texto CON Ácentos y Múltiples   Espacios!!  ": "texto con acentos y multiples espacios",
        "CAPÓ DELANTERO": "capo delantero",
        "Lámpara Trasera Izquierda (Stop)": "lampara trasera izquierda stop",
        "Rin 17\" Lujo": "rin 17 lujo",
        "GUARDABARRO DEL. DER.": "guardabarro del der",
        "Puerta Delantera Izquierda": "puerta delantera izquierda",
        "  Múltiples   \t\n   espacios  ": "multiples espacios",
        "SinPuntuaciónNiAcentos": "sinpuntuacionniacentos",
        "Número123": "numero123",
        "BOMPER TRASERO C/HUECOS SENSORES": "bomper trasero chuecos sensores",  # c/ is removed
        "Fáröla Ízquíérdá": "farola izquierda"
    }

    for original, expected in test_strings.items():
        normalized = normalize_text(original)
        print(f"Original: '{original}'")
        print(f"Normalized: '{normalized}'")
        print(f"Expected:   '{expected}'")
        print(f"Match: {normalized == expected}\n")

    # Test with None or non-string
    print(f"Normalizing None: '{normalize_text(None)}'")
    # print(f"Normalizing a number (123): '{normalize_text(123)}'") # This would error if not handled
