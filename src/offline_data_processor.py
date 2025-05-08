import json
import sqlite3
import pandas as pd
import argparse
import os
from utils.text_utils import normalize_text

# --- Configuration ---
# Default paths (can be overridden by command-line arguments)
# These paths are relative to the directory where the script is executed from.
# Assuming execution from 'Source_Files/' (e.g., python src/offline_data_processor.py)
DEFAULT_EQUIVALENCIAS_PATH = "Equivalencias.xlsx"
# Using the cleaned version for initial processing as it's smaller.
# The script can also handle the original Consolidado.json or its chunks if memory allows.
DEFAULT_CONSOLIDADO_PATH = "Consolidado_cleaned.json"
# Will be created in Source_Files/data/
DEFAULT_DB_PATH = "data/fixacar_history.db"

# --- Helper Functions ---


def load_equivalencias(file_path: str) -> dict:
    """
    Loads Equivalencias.xlsx, normalizes terms, and creates a mapping
    from normalized terms to a generated Equivalencia_Row_ID (1-based row index).
    (Corresponds to TODO Task 1.3, updated for new Excel structure)
    """
    print(f"Loading equivalencias from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: Equivalencias file not found at {file_path}")
        return {}

    try:
        # Read the Excel file. Pandas will use the first row as headers by default.
        # The headers are Column1, Column2, etc.
        df = pd.read_excel(file_path, sheet_name=0)
        equivalencias_map = {}

        # Iterate through rows, using 1-based index for Equivalencia_Row_ID
        for index, row in df.iterrows():
            equivalencia_row_id = index + 1  # 1-based ID

            # Iterate through all columns in the current row
            # df.columns will be ['Column1', 'Column2', ...]
            for col_name in df.columns:
                term = row[col_name]
                # Check if cell is not NaN and not empty string
                if pd.notna(term) and str(term).strip():
                    normalized_term = normalize_text(str(term))
                    if normalized_term:  # Ensure not empty after normalization
                        # If a normalized term could belong to multiple groups (rows),
                        # this will overwrite with the latest one.
                        # If a term appears in multiple rows, the last row's ID will be used.
                        # This behavior should be acceptable as per PRD (one term maps to one ID).
                        equivalencias_map[normalized_term] = equivalencia_row_id

        print(
            f"Loaded {len(equivalencias_map)} normalized term mappings from {len(df)} rows in Equivalencias.")
        return equivalencias_map
    except Exception as e:
        print(f"Error loading or processing Equivalencias.xlsx: {e}")
        return {}


def setup_database(db_path: str) -> sqlite3.Connection | None:
    """
    Sets up the SQLite database and the historical_parts table.
    (Corresponds to TODO Task 1.9)
    """
    print(f"Setting up database at: {db_path}")
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_parts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vin_number TEXT,
            vin_make TEXT,
            vin_model TEXT,
            vin_year INTEGER,
            vin_series TEXT,
            vin_bodystyle TEXT,
            original_description TEXT,
            normalized_description TEXT,
            sku TEXT,
            Equivalencia_Row_ID INTEGER,
            source_bid_id TEXT, 
            UNIQUE(vin_number, original_description, sku) -- Basic uniqueness constraint
        )
        ''')
        conn.commit()
        print("Database and 'historical_parts' table created/ensured.")
        return conn
    except Exception as e:
        print(f"Error setting up database: {e}")
        if conn:
            conn.close()
        return None


def process_consolidado_data(conn: sqlite3.Connection, consolidado_path: str, equivalencias_map: dict):
    """
    Processes Consolidado.json, filters, normalizes, links, and inserts data into SQLite.
    (Corresponds to TODO Tasks 1.4, 1.5, 1.6, 1.7, 1.8, 1.10)
    """
    print(f"Processing consolidado data from: {consolidado_path}")
    if not os.path.exists(consolidado_path):
        print(f"Error: Consolidado file not found at {consolidado_path}")
        return

    # This script is designed for the full Consolidado.json.
    # If it's too large for memory, it should process chunk files instead.
    # For now, assuming it can handle the target file or a representative large chunk.

    processed_count = 0
    inserted_count = 0
    skipped_no_sku = 0
    skipped_duplicate = 0

    try:
        with open(consolidado_path, 'r', encoding='utf-8') as f:
            # For very large JSON, a streaming parser would be better.
            # json.load() loads the whole file, which might cause MemoryError.
            # If Consolidado.json is an array of objects at the root:
            all_records = json.load(f)

        cursor = conn.cursor()

        for record in all_records:
            processed_count += 1
            if processed_count % 1000 == 0:
                print(
                    f"Processed {processed_count} records from consolidado...")

            vin_number = record.get("vin_number")
            # From PRD, assuming 'maker' maps to 'vin_make'
            vin_make = record.get("maker")
            # Assuming 'model' from JSON maps to 'vin_model'
            vin_model = record.get("model")
            # Assuming 'fabrication_year'
            vin_year_str = record.get("fabrication_year")
            vin_series = record.get("series")
            # vin_bodystyle is not directly in the sample, might need to infer or leave NULL
            # Placeholder, adjust if field name differs
            vin_bodystyle = record.get("body_style")

            try:
                vin_year = int(vin_year_str) if vin_year_str else None
            except (ValueError, TypeError):
                vin_year = None

            # Assuming 'quote' is the bid ID
            source_bid_id = record.get("quote")

            items = record.get("items", [])
            if not isinstance(items, list):
                items = []

            for item_detail in items:
                if not isinstance(item_detail, dict):
                    continue

                sku = item_detail.get("referencia")
                original_description = item_detail.get("descripcion")

                if not sku:  # Task 1.5: Filter for non-empty SKUs
                    skipped_no_sku += 1
                    continue

                if not original_description:  # Skip if no description
                    continue

                normalized_description = normalize_text(original_description)

                # Task 1.8: Look up Equivalencia_Row_ID
                equivalencia_row_id = equivalencias_map.get(
                    normalized_description)  # Can be None

                # Task 1.10: Load into SQLite
                try:
                    cursor.execute('''
                    INSERT INTO historical_parts (
                        vin_number, vin_make, vin_model, vin_year, vin_series, vin_bodystyle,
                        original_description, normalized_description, sku, Equivalencia_Row_ID, source_bid_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        vin_number, vin_make, vin_model, vin_year, vin_series, vin_bodystyle,
                        original_description, normalized_description, sku, equivalencia_row_id, source_bid_id
                    ))
                    inserted_count += 1
                except sqlite3.IntegrityError:  # Handles UNIQUE constraint violation
                    skipped_duplicate += 1
                except Exception as e_insert:
                    print(
                        f"Error inserting row: {e_insert} for VIN {vin_number}, SKU {sku}")

        conn.commit()
        print(
            f"Finished processing consolidado. Total records processed: {processed_count}")
        print(f"Items inserted into database: {inserted_count}")
        print(f"Items skipped (no SKU): {skipped_no_sku}")
        print(f"Items skipped (duplicate): {skipped_duplicate}")

    except FileNotFoundError:
        print(f"Error: Consolidado file not found at {consolidado_path}")
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {consolidado_path}. File might be corrupted or not valid JSON.")
    except MemoryError:
        print(
            f"Error: MemoryError while loading {consolidado_path}. The file is too large for this script's current approach.")
        print("Consider modifying this script to process chunk files (e.g., Consolidado_chunk_*.json) one by one.")
    except Exception as e:
        print(
            f"An unexpected error occurred while processing {consolidado_path}: {e}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Offline data processor for Fixacar SKU Finder.")
    parser.add_argument("--equivalencias", default=DEFAULT_EQUIVALENCIAS_PATH,
                        help="Path to Equivalencias.xlsx file.")
    parser.add_argument("--consolidado", default=DEFAULT_CONSOLIDADO_PATH,
                        help="Path to Consolidado.json file (or a large chunk).")
    parser.add_argument("--dbpath", default=DEFAULT_DB_PATH,
                        help="Path to the SQLite database file to create/update.")
    args = parser.parse_args()

    print("--- Starting Offline Data Processing ---")

    # 1. Load Equivalencias (Task 1.3)
    equivalencias_map = load_equivalencias(args.equivalencias)
    if not equivalencias_map:
        print("Failed to load equivalencias. Aborting.")
        return

    # 2. Setup Database (Task 1.9)
    conn = setup_database(args.dbpath)
    if not conn:
        print("Failed to setup database. Aborting.")
        return

    # 3. Process Consolidado and Load into DB (Tasks 1.4-1.8, 1.10)
    #    This function will handle reading Consolidado.json and inserting.
    #    IMPORTANT: If Consolidado.json is too large, this part needs to be adapted
    #    to read the smaller _chunk_ files generated by split_json.py.
    #    For now, it assumes args.consolidado points to a file that can be loaded.
    #    The user might need to point it to one of the _cleaned_ files if the original is too big.
    #    Or, ideally, this script would iterate through all Consolidado_chunk_*.json files.
    #    Let's modify it to iterate through chunks if the main file is too big or not found.

    # Check if the main consolidado file exists and is not empty
    consolidado_file_to_process = args.consolidado

    # Modification: If the main Consolidado.json is too large (as indicated by previous steps),
    # this script should ideally process the individual chunks.
    # For now, it will try the specified file. If it fails with MemoryError,
    # the user would need to run it on smaller files or this script would need modification.
    # A more robust version would loop through Consolidado_chunk_1.json to Consolidado_chunk_10.json.
    # Let's make a simple adjustment to process chunks if the main file is named "Consolidado.json"
    # and it's likely too big.

    # For this iteration, we will assume the user provides a manageable file (e.g. Consolidado_cleaned.json or a single chunk)
    # or that the full Consolidado.json can be handled. A future step could be to make this
    # script iterate over the 10 main chunks.

    process_consolidado_data(
        conn, consolidado_file_to_process, equivalencias_map)

    if conn:
        conn.close()

    print("--- Offline Data Processing Finished ---")


if __name__ == "__main__":
    # To run this script from the Source_Files directory:
    # python src/offline_data_processor.py
    # Or with custom paths:
    # python src/offline_data_processor.py --equivalencias ../Equivalencias.xlsx --consolidado ../Consolidado_cleaned.json --dbpath data/fixacar_history.db
    main()
