"""
PyInstaller runtime hook to fix spaCy model loading in packaged executables.
This hook ensures that the Spanish model es_core_news_sm can be found.
"""

import os
import sys

def fix_spacy_model_paths():
    """Fix spaCy model paths for packaged executables"""
    try:
        # Only run this fix when running as a packaged executable
        if not getattr(sys, 'frozen', False):
            return
            
        # Get the temporary directory where PyInstaller extracts files
        if hasattr(sys, '_MEIPASS'):
            temp_dir = sys._MEIPASS
            
            # Add potential spaCy model locations to sys.path
            model_paths = [
                os.path.join(temp_dir, 'es_core_news_sm'),
                os.path.join(temp_dir, 'spacy', 'data', 'es_core_news_sm'),
                os.path.join(temp_dir, 'lib', 'python3.11', 'site-packages', 'es_core_news_sm'),
            ]
            
            for path in model_paths:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)
                    print(f"✅ Added spaCy model path: {path}")
            
            # Try to register the model with spaCy
            try:
                import spacy
                from spacy.util import find_data_path
                
                # Override spaCy's model finding mechanism
                original_find_data_path = find_data_path
                
                def patched_find_data_path(name):
                    """Patched version that looks in our bundled locations"""
                    if name == 'es_core_news_sm':
                        for path in model_paths:
                            if os.path.exists(path):
                                print(f"✅ Found spaCy model at: {path}")
                                return path
                    return original_find_data_path(name)
                
                # Monkey patch the function
                spacy.util.find_data_path = patched_find_data_path
                
                print("✅ spaCy model path fix applied successfully")
                
            except ImportError as e:
                print(f"⚠️ Could not apply spaCy model fix: {e}")
                
    except Exception as e:
        print(f"⚠️ spaCy model path fix failed: {e}")

# Apply the fix when this module is imported
fix_spacy_model_paths()
