#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get_New_Data_From_Json.py

This script compares two large JSON files containing data collected over time by a bot:
- Original file: "Source_Files/Consolidado.json" in the SKU_Predictor_v2.0 project
- New file: "C:/Users/juanp/Downloads/Nuevo_Consolidado.json"

It identifies and extracts only the new reports/data that exist in the new JSON file but not in the original file,
saves the extracted new data to a new JSON file named "New_Data.json", and creates a SQLite database file
from the New_Data.json content, following the same structure as the existing database in the project,
but keeping the dates of each record.

Author: Augment Agent
Date: 2024
"""

import json
import sqlite3
import os
import sys
import logging
import re
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("get_new_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Get_New_Data_From_Json")

# File paths
ORIGINAL_JSON_PATH = "Source_Files/Consolidado.json"
NEW_JSON_PATH = "C:/Users/juanp/Downloads/Nuevo_Consolidado.json"
OUTPUT_JSON_PATH = "New_Data.json"
OUTPUT_DB_PATH = "New_Data.db"

# Base paths for project
PROJECT_BASE_PATH = "C:/Users/juanp/OneDrive/Documents/Python/0_Training/017_Fixacar/010_SKU_Predictor_v2.0"


def setup_database(db_path: str) -> Optional[sqlite3.Connection]:
    """
    Sets up the SQLite database with the same structure as the existing database.

    Args:
        db_path: Path to the SQLite database file to create

    Returns:
        SQLite connection object or None if setup fails
    """
    logger.info(f"Setting up database at: {db_path}")
    conn = None

    try:
        # Only create directory if db_path has a directory component
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create filtered_bids table (same structure as in the existing database)
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
            item_sku TEXT NOT NULL,
            date TEXT
        )
        ''')

        # Create historical_parts table (same structure as in the existing database)
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
            date TEXT,
            UNIQUE(vin_number, original_description, sku)
        )
        ''')

        conn.commit()
        logger.info("Database and tables created successfully.")
        return conn

    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        if conn:
            conn.close()
        return None


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSON file and returns its contents.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of dictionaries containing the JSON data
    """
    logger.info(f"Loading JSON file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(
            f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return []


def normalize_text(text: str) -> str:
    """
    Normalizes text by removing special characters, extra spaces, and converting to lowercase.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Replace common Spanish abbreviations and contractions
    text = re.sub(r'izq\.?', 'izquierdo', text)
    text = re.sub(r'der\.?', 'derecho', text)
    text = re.sub(r'sup\.?', 'superior', text)
    text = re.sub(r'inf\.?', 'inferior', text)
    text = re.sub(r'del\.?', 'delantero', text)
    text = re.sub(r'tras\.?', 'trasero', text)

    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace special chars with space
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_new_records(original_data: List[Dict[str, Any]], new_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compares original and new data to extract only the new records.

    Args:
        original_data: List of dictionaries containing the original data
        new_data: List of dictionaries containing the new data

    Returns:
        List of dictionaries containing only the new records
    """
    logger.info("Extracting new records...")

    # Create a set of quote IDs from the original data for faster lookup
    original_quote_ids = {record.get('quote')
                          for record in original_data if record.get('quote')}

    # Extract records that exist in the new data but not in the original data
    new_records = [record for record in new_data if record.get(
        'quote') and record.get('quote') not in original_quote_ids]

    logger.info(f"Found {len(new_records)} new records")
    return new_records


def save_json_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """
    Saves data to a JSON file.

    Args:
        data: List of dictionaries to save
        file_path: Path to save the JSON file

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Saving {len(data)} records to {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False


def insert_data_into_db(conn: sqlite3.Connection, data: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Inserts data into the SQLite database (both filtered_bids and historical_parts tables).

    Args:
        conn: SQLite connection object
        data: List of dictionaries containing the data to insert

    Returns:
        Tuple of (filtered_bids_count, historical_parts_count) records inserted
    """
    logger.info(f"Inserting {len(data)} records into database...")

    filtered_bids_count = 0
    historical_parts_count = 0
    skipped_no_sku = 0
    skipped_duplicate = 0

    try:
        cursor = conn.cursor()

        for record in data:
            # Extract top-level bid information
            vin_number = record.get("vin_number")
            vin_make = record.get("maker")
            vin_model = record.get("model")
            vin_year_str = record.get("fabrication_year")
            vin_series = record.get("series")
            vin_bodystyle = record.get("body_style")
            source_bid_id = record.get("quote")

            # Extract date information
            date_obj = record.get("date", {}).get("$date")
            date_str = None
            if date_obj:
                try:
                    # Convert ISO format date to string
                    date_str = date_obj
                except Exception as e:
                    logger.warning(f"Error parsing date {date_obj}: {e}")

            try:
                vin_year = int(vin_year_str) if vin_year_str else None
            except (ValueError, TypeError):
                vin_year = None

            # Process items within the record
            items = record.get("items", [])
            if not isinstance(items, list):
                items = []

            for item in items:
                if not isinstance(item, dict):
                    continue

                # Check for both possible field names (English and Spanish)
                original_description = item.get("description", "").strip()
                if not original_description:
                    original_description = item.get("descripcion", "").strip()

                sku = item.get("sku", "").strip()
                if not sku:
                    sku = item.get("referencia", "").strip()

                # Filter: Keep only if SKU exists and is not just whitespace
                if sku:
                    # Insert into filtered_bids table
                    try:
                        cursor.execute('''
                        INSERT INTO filtered_bids (
                            vin_number, vin_make, vin_model, vin_year, vin_series, vin_bodystyle,
                            source_bid_id, item_original_description, item_sku, date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            vin_number, vin_make, vin_model, vin_year, vin_series, vin_bodystyle,
                            source_bid_id, original_description, sku, date_str
                        ))
                        filtered_bids_count += 1
                    except Exception as e_insert:
                        logger.error(
                            f"Error inserting row into filtered_bids: {e_insert} for VIN {vin_number}, SKU {sku}")

                    # Insert into historical_parts table
                    try:
                        # Normalize the description
                        normalized_description = normalize_text(
                            original_description)

                        # For now, set Equivalencia_Row_ID to NULL (would need to load from Equivalencias.xlsx)
                        equivalencia_row_id = None

                        cursor.execute('''
                        INSERT INTO historical_parts (
                            vin_number, vin_make, vin_model, vin_year, vin_series, vin_bodystyle,
                            original_description, normalized_description, sku, Equivalencia_Row_ID, source_bid_id, date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            vin_number, vin_make, vin_model, vin_year, vin_series, vin_bodystyle,
                            original_description, normalized_description, sku, equivalencia_row_id, source_bid_id, date_str
                        ))
                        historical_parts_count += 1
                    except sqlite3.IntegrityError:
                        # Handle UNIQUE constraint violation
                        skipped_duplicate += 1
                    except Exception as e_insert:
                        logger.error(
                            f"Error inserting row into historical_parts: {e_insert} for VIN {vin_number}, SKU {sku}")
                else:
                    skipped_no_sku += 1

        conn.commit()
        logger.info(
            f"Successfully inserted {filtered_bids_count} records into filtered_bids table")
        logger.info(
            f"Successfully inserted {historical_parts_count} records into historical_parts table")
        logger.info(f"Skipped {skipped_no_sku} items with no SKU")
        logger.info(
            f"Skipped {skipped_duplicate} duplicate items in historical_parts table")

        return (filtered_bids_count, historical_parts_count)

    except Exception as e:
        logger.error(f"Error inserting data into database: {e}")
        return (0, 0)


def main():
    """Main function to execute the script."""
    logger.info("Starting Get_New_Data_From_Json.py")

    # Step 1: Load the original JSON file
    original_data = load_json_file(os.path.join(
        PROJECT_BASE_PATH, ORIGINAL_JSON_PATH))
    if not original_data:
        logger.error("Failed to load original data. Exiting.")
        return

    # Step 2: Load the new JSON file
    new_data = load_json_file(NEW_JSON_PATH)
    if not new_data:
        logger.error("Failed to load new data. Exiting.")
        return

    # Step 3: Extract new records
    new_records = extract_new_records(original_data, new_data)
    if not new_records:
        logger.info("No new records found. Exiting.")
        return

    # Step 4: Save new records to JSON file
    if not save_json_file(new_records, OUTPUT_JSON_PATH):
        logger.error("Failed to save new records to JSON file. Exiting.")
        return

    # Step 5: Set up SQLite database
    conn = setup_database(OUTPUT_DB_PATH)
    if not conn:
        logger.error("Failed to set up database. Exiting.")
        return

    # Step 6: Insert data into database
    # We don't need to use the counts here as we'll query the database directly
    _ = insert_data_into_db(conn, new_records)

    # Step 7: Close database connection
    if conn:
        try:
            # Final check on row counts
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM filtered_bids")
            filtered_row_count = cursor.fetchone()[0]
            logger.info(
                f"Final row count in 'filtered_bids': {filtered_row_count}")

            cursor.execute("SELECT COUNT(*) FROM historical_parts")
            historical_row_count = cursor.fetchone()[0]
            logger.info(
                f"Final row count in 'historical_parts': {historical_row_count}")
        except Exception as e:
            logger.error(f"Error querying final row count: {e}")
        finally:
            conn.close()
            logger.info("Database connection closed.")

    logger.info("Get_New_Data_From_Json.py completed successfully")


if __name__ == "__main__":
    main()
