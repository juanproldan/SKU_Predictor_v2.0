import tkinter as tk
from tkinter import ttk
import os
import pandas as pd
# Assuming text_utils is in the same directory or src is in PYTHONPATH
try:
    from utils.text_utils import normalize_text
except ImportError:
    # Fallback for direct execution if src is not in path (e.g. during development)
    from .utils.text_utils import normalize_text


# --- Configuration ---
DEFAULT_EQUIVALENCIAS_PATH = "Equivalencias.xlsx"
# Will be created in Source_Files/data/
DEFAULT_MAESTRO_PATH = "data/Maestro.xlsx"
DEFAULT_DB_PATH = "data/fixacar_history.db"

# In-memory data stores
equivalencias_map_global = {}
maestro_data_global = []


class FixacarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fixacar SKU Finder v1.0")
        self.root.geometry("800x600")

        # Load initial data
        self.load_all_data()

        # Setup UI
        self.create_widgets()

    def load_all_data(self):
        """Loads all necessary data on startup."""
        global equivalencias_map_global, maestro_data_global

        print("--- Loading Application Data ---")
        equivalencias_map_global = self.load_equivalencias_data(
            DEFAULT_EQUIVALENCIAS_PATH)
        maestro_data_global = self.load_maestro_data(
            DEFAULT_MAESTRO_PATH, equivalencias_map_global)
        # DB connection will be established when needed for search
        print("--- Application Data Loading Complete ---")

    def load_equivalencias_data(self, file_path: str) -> dict:
        """
        Loads Equivalencias.xlsx, normalizes terms, and creates a mapping
        from normalized terms to a generated Equivalencia_Row_ID (1-based row index).
        (Corresponds to TODO Task 2.4)
        """
        print(f"Loading equivalencias from: {file_path}")
        if not os.path.exists(file_path):
            print(
                f"Warning: Equivalencias file not found at {file_path}. Equivalency linking will be disabled.")
            return {}

        try:
            df = pd.read_excel(file_path, sheet_name=0)
            equivalencias_map = {}
            for index, row in df.iterrows():
                equivalencia_row_id = index + 1  # 1-based ID
                for col_name in df.columns:
                    term = row[col_name]
                    if pd.notna(term) and str(term).strip():
                        normalized_term = normalize_text(str(term))
                        if normalized_term:
                            equivalencias_map[normalized_term] = equivalencia_row_id

            print(
                f"Loaded {len(equivalencias_map)} normalized term mappings from {len(df)} rows in Equivalencias.")
            return equivalencias_map
        except Exception as e:
            print(
                f"Error loading or processing Equivalencias.xlsx: {e}. Equivalency linking will be disabled.")
            return {}

    def load_maestro_data(self, file_path: str, equivalencias_map: dict) -> list:
        """
        Loads Maestro.xlsx into an in-memory list of dictionaries.
        Creates the file with headers if it doesn't exist.
        Normalizes descriptions and links Equivalencia_Row_ID.
        (Corresponds to TODO Task 2.5)
        """
        print(f"Loading maestro data from: {file_path}")
        maestro_columns = [
            'Maestro_ID', 'VIN_Make', 'VIN_Model', 'VIN_Year_Min', 'VIN_Year_Max',
            'VIN_Series_Trim', 'VIN_BodyStyle', 'Original_Description_Input',
            'Normalized_Description_Input', 'Equivalencia_Row_ID', 'Confirmed_SKU',
            'Confidence', 'Source', 'Date_Added'
        ]

        # Ensure data directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            print(
                f"Maestro file not found at {file_path}. Creating a new one with headers.")
            df_maestro = pd.DataFrame(columns=maestro_columns)
            try:
                df_maestro.to_excel(file_path, index=False)
                return []  # Return empty list for a new file
            except Exception as e:
                print(
                    f"Error creating Maestro.xlsx: {e}. Maestro data will be empty.")
                return []

        try:
            df_maestro = pd.read_excel(file_path, sheet_name=0)
            if df_maestro.empty:
                print("Maestro file is empty.")
                return []

            maestro_list = []
            for _, row in df_maestro.iterrows():
                entry = row.to_dict()
                # Normalize Original_Description_Input if it exists and store it
                original_desc = entry.get('Original_Description_Input', "")
                if pd.notna(original_desc):
                    normalized_desc = normalize_text(str(original_desc))
                    entry['Normalized_Description_Input'] = normalized_desc
                    # Link Equivalencia_Row_ID
                    entry['Equivalencia_Row_ID'] = equivalencias_map.get(
                        normalized_desc)
                else:
                    entry['Normalized_Description_Input'] = ""
                    entry['Equivalencia_Row_ID'] = None

                maestro_list.append(entry)

            print(f"Loaded {len(maestro_list)} records from Maestro.xlsx.")
            return maestro_list
        except Exception as e:
            print(
                f"Error loading or processing Maestro.xlsx: {e}. Maestro data will be empty.")
            return []

    def create_widgets(self):
        """Creates the GUI widgets."""
        # Placeholder for UI elements
        label = ttk.Label(
            self.root, text="Fixacar SKU Finder - UI Placeholder")
        label.pack(padx=10, pady=10)

        # More UI elements will be added in Phase 3 (VIN input, Descriptions, Button)
        # and Phase 5 (Output area, selections)


if __name__ == '__main__':
    # This allows running the script directly from Source_Files/
    # For utils.text_utils to be found, ensure src is in PYTHONPATH or run as module if needed
    # e.g. python -m src.main_app from Source_Files parent.
    # For simplicity now, direct run from Source_Files is assumed with try-except for import.

    # If running from Source_Files directory:
    # Make sure Equivalencias.xlsx is in Source_Files
    # Maestro.xlsx will be created in Source_Files/data/
    # fixacar_history.db should be in Source_Files/data/ (created by offline_processor)

    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print(
        f"Expected Equivalencias.xlsx at: {os.path.join(current_dir, DEFAULT_EQUIVALENCIAS_PATH)}")
    print(
        f"Expected/Creating Maestro.xlsx at: {os.path.join(current_dir, DEFAULT_MAESTRO_PATH)}")
    print(
        f"Expected fixacar_history.db at: {os.path.join(current_dir, DEFAULT_DB_PATH)}")

    root = tk.Tk()
    app = FixacarApp(root)
    root.mainloop()
