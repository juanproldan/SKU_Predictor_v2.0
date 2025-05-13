import json
import sqlite3
import argparse
import os
import sys

# --- Configuration ---
# Default paths (relative to script execution directory)
# Assuming execution from 'Source_Files/' (e.g., python src/create_consolidado_db.py)
DEFAULT_CONSOLIDADO_PATH = "Consolidado_cleaned.json"  # Use cleaned by default
DEFAULT_DB_PATH = "data/consolidado.db"  # Place DB in data subfolder

# --- Helper Functions ---


def setup_database(db_path: str) -> sqlite3.Connection | None:
    """
    Sets up the SQLite database and the filtered_bids table.
    """
    print(f"Setting up database at: {db_path}")
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS filtered_bids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vin_number TEXT,
            vin_make TEXT,
            vin_model TEXT,
            vin_year INTEGER,
            vin_series TEXT,
            vin_bodystyle TEXT,
            source_bid_id TEXT,
            item_original_description TEXT,
            item_sku TEXT NOT NULL -- Ensure SKU is not NULL in the DB itself
        )
        ''')
        # Optional: Add index for faster SKU lookups if needed later
        # cursor.execute('CREATE INDEX IF NOT EXISTS idx_sku ON filtered_bids (item_sku)')
        conn.commit()
        print("Database and 'filtered_bids' table created/ensured.")
        return conn
    except Exception as e:
        print(f"Error setting up database: {e}", file=sys.stderr)
        if conn:
            conn.close()
        return None


def process_consolidado_to_db(conn: sqlite3.Connection, consolidado_path: str):
    """
    Reads Consolidado.json, filters items based on non-empty SKU,
    and inserts filtered data into the SQLite database.
    """
    print(f"Processing consolidado data from: {consolidado_path}")
    if not os.path.exists(consolidado_path):
        print(
            f"Error: Consolidado file not found at {consolidado_path}", file=sys.stderr)
        return

    processed_records = 0
    processed_items = 0
    inserted_rows = 0
    skipped_items_no_sku = 0

    try:
        with open(consolidado_path, 'r', encoding='utf-8') as f:
            # Load the entire JSON. Consider streaming for very large files.
            all_records = json.load(f)

        cursor = conn.cursor()

        for record in all_records:
            processed_records += 1
            if processed_records % 1000 == 0:
                print(f"Processed {processed_records} records...")

            # Extract top-level bid information
            vin_number = record.get("vin_number")
            vin_make = record.get("maker")
            vin_model = record.get("model")
            vin_year_str = record.get("fabrication_year")
            vin_series = record.get("series")
            # Adjust if field name differs
            vin_bodystyle = record.get("body_style")
            source_bid_id = record.get("quote")

            try:
                vin_year = int(vin_year_str) if vin_year_str else None
            except (ValueError, TypeError):
                vin_year = None

            # Process items within the record
            items = record.get("items", [])
            if not isinstance(items, list):
                items = []

            for item_detail in items:
                processed_items += 1
                if not isinstance(item_detail, dict):
                    continue

                sku = item_detail.get("referencia")
                original_description = item_detail.get("descripcion")

                # Filter: Keep only if SKU exists and is not just whitespace
                if sku and str(sku).strip():
                    valid_sku = str(sku).strip()  # Use the stripped version
                    try:
                        cursor.execute('''
                        INSERT INTO filtered_bids (
                            vin_number, vin_make, vin_model, vin_year, vin_series, vin_bodystyle,
                            source_bid_id, item_original_description, item_sku
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            vin_number, vin_make, vin_model, vin_year, vin_series, vin_bodystyle,
                            source_bid_id, original_description, valid_sku
                        ))
                        inserted_rows += 1
                    except Exception as e_insert:
                        # Catch potential DB errors during insert
                        print(
                            f"Error inserting row: {e_insert} for VIN {vin_number}, SKU {valid_sku}", file=sys.stderr)
                else:
                    skipped_items_no_sku += 1

        conn.commit()
        print("\n--- Processing Summary ---")
        print(f"Total records processed from JSON: {processed_records}")
        print(f"Total items processed: {processed_items}")
        print(f"Items skipped (missing/empty SKU): {skipped_items_no_sku}")
        print(f"Rows inserted into 'filtered_bids' table: {inserted_rows}")

    except FileNotFoundError:
        print(
            f"Error: Consolidado file not found at {consolidado_path}", file=sys.stderr)
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {consolidado_path}. File might be corrupted.", file=sys.stderr)
    except MemoryError:
        print(
            f"Error: MemoryError loading {consolidado_path}. File may be too large.", file=sys.stderr)
        print("Consider using a smaller chunk file or modifying the script for streaming.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Cleans Consolidado.json by filtering for items with valid SKUs and saves to an SQLite DB.")
    parser.add_argument("--consolidado", default=DEFAULT_CONSOLIDADO_PATH,
                        help=f"Path to the input Consolidado JSON file (default: {DEFAULT_CONSOLIDADO_PATH}).")
    parser.add_argument("--dbpath", default=DEFAULT_DB_PATH,
                        help=f"Path to the output SQLite database file (default: {DEFAULT_DB_PATH}).")
    args = parser.parse_args()

    print("--- Starting Consolidado to DB Processing ---")
    print(f"Input JSON: {args.consolidado}")
    print(f"Output DB: {args.dbpath}")

    # 1. Setup Database
    conn = setup_database(args.dbpath)
    if not conn:
        print("Failed to setup database. Aborting.", file=sys.stderr)
        return

    # 2. Process JSON and Load into DB
    process_consolidado_to_db(conn, args.consolidado)

    # 3. Close DB connection
    if conn:
        try:
            # Final check on row count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM filtered_bids")
            row_count = cursor.fetchone()[0]
            print(f"Final row count in 'filtered_bids': {row_count}")
        except Exception as e:
            print(f"Error querying final row count: {e}", file=sys.stderr)
        finally:
            conn.close()
            print("Database connection closed.")

    print("--- Consolidado to DB Processing Finished ---")


if __name__ == "__main__":
    # Example usage from the 'Source_Files' directory:
    # python src/create_consolidado_db.py
    #
    # Example with custom paths:
    # python src/create_consolidado_db.py --consolidado ../Consolidado_Full.json --dbpath ../data/full_consolidado_filtered.db
    main()
