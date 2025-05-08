import unicodedata
import re


def normalize_text(text: str) -> str:
    """
    Normalizes a text string according to the project's requirements:
    - Converts to lowercase.
    - Removes leading/trailing whitespace.
    - Standardizes internal whitespace (multiple spaces to one).
    - Removes common punctuation (keeps alphanumeric and spaces).
    - Handles accented characters/diacritics (e.g., converts 'á' to 'a').
    """
    if not isinstance(text, str):
        return ""  # Or raise an error, depending on desired handling for non-strings

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

    return text


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
