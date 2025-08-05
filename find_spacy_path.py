import spacy
import os

try:
    # Get the package path for the model
    model_path = os.path.dirname(spacy.load('es_core_news_sm').path)
    print(f"spaCy model data path found: {model_path}")
except OSError:
    print("Error: The model 'es_core_news_sm' is not downloaded.")
    print("Please run: python -m spacy download es_core_news_sm")
