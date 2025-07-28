#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Consolidado Processor

This script processes Consolidado.json into a single processed_consolidado.db database
that serves both VIN training and SKU prediction needs. It preserves all useful records
while cleaning VINs and normalizing text descriptions.

Key Features:
- Reads Consolidado.json from Source_Files/
- Cleans VINs using validation logic (preserves records with invalid VINs)
- Processes text using normalization rules
- Creates single processed_consolidado.db
- Supports both VIN training and SKU prediction needs
- Uses dynamic path resolution for client deployment

Author: Augment Agent
Date: 2025-07-24
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import re
from pathlib import Path
import logging
from datetime import datetime

# Add utils to path for text processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.text_utils import normalize_text

# --- Configuration ---
def get_base_path():
    """Get the base path for the application, works for both script and executable."""
    if getattr(sys, 'frozen', False):
        # Running as executable - use executable's directory
        return os.path.dirname(sys.executable)
    else:
        # Running as script - use project root
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = get_base_path()
SOURCE_FILES_DIR = os.path.join(BASE_PATH, "Source_Files")
LOGS_DIR = os.path.join(BASE_PATH, "logs")

# File paths - everything in Source_Files for unified structure
CONSOLIDADO_PATH = os.path.join(SOURCE_FILES_DIR, "Consolidado.json")
TEXT_PROCESSING_PATH = os.path.join(SOURCE_FILES_DIR, "Text_Processing_Rules.xlsx")
OUTPUT_DB_PATH = os.path.join(SOURCE_FILES_DIR, "processed_consolidado.db")
LOG_PATH = os.path.join(LOGS_DIR, f"consolidado_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# --- Logging Setup ---
def setup_logging():
    """Setup logging configuration."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# --- VIN Cleaning Functions ---
def validate_vin_format(vin_str):
    """
    Basic VIN format validation.
    Returns True if VIN has correct format (17 alphanumeric characters).
    """
    if not vin_str or len(vin_str) != 17:
        return False
    
    # Check for valid characters (no I, O, Q allowed in VINs)
    invalid_chars = set('IOQ')
    if any(char in invalid_chars for char in vin_str.upper()):
        return False
    
    # Must be alphanumeric
    if not vin_str.isalnum():
        return False
    
    return True

def validate_vin_check_digit(vin_str):
    """
    Validate VIN check digit (position 9) using the standard algorithm.
    This is optional validation - some VINs may have incorrect check digits
    but still be valid for maker/fabrication_year/series extraction.
    """
    # VIN character values for check digit calculation
    char_values = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9, 'S': 2,
        'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
    }

    # Position weights for check digit calculation
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

    try:
        total = sum(char_values[char] * weight for char, weight in zip(vin_str, weights))
        calculated_check = total % 11
        expected_check = 'X' if calculated_check == 10 else str(calculated_check)
        return vin_str[8] == expected_check
    except (KeyError, IndexError):
        return False

def clean_vin_for_training(vin):
    """
    Clean and validate VIN for training purposes.
    Returns cleaned VIN if valid, None if invalid.
    """
    if not vin:
        return None
    
    # Convert to string and clean
    vin_str = str(vin).strip().upper()
    
    # Basic format validation
    if not validate_vin_format(vin_str):
        return None
    
    # Optional: Check digit validation (commented out as it may be too strict)
    # if not validate_vin_check_digit(vin_str):
    #     return None
    
    return vin_str

# --- Database Setup ---
def setup_database(db_path):
    """
    Sets up the SQLite database and the processed_consolidado table.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up database at: {db_path}")

    try:
        # Ensure Source_Files directory exists (unified structure)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the unified table - SIMPLIFIED SCHEMA (removed 5 unnecessary columns)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_consolidado (
            vin_number TEXT,                    -- Cleaned VINs for VIN training (may be NULL)
            maker TEXT,                         -- For both VIN & SKU training
            fabrication_year INTEGER,          -- For both VIN & SKU training
            series TEXT,                        -- For both VIN & SKU training
            original_descripcion TEXT,          -- For SKU training (may be NULL)
            normalized_descripcion TEXT,        -- For SKU training, processed (may be NULL)
            referencia TEXT,                    -- For SKU training (may be NULL)
            UNIQUE(vin_number, original_descripcion, referencia) -- Prevent duplicates
        )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vin_training ON processed_consolidado (vin_number, maker, fabrication_year, series)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sku_training ON processed_consolidado (maker, fabrication_year, series, referencia)')
        
        conn.commit()
        logger.info("Database and 'processed_consolidado' table created/ensured.")
        return conn
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        if 'conn' in locals():
            conn.close()
        return None

# --- Text Processing ---
def load_equivalencias_map(text_processing_path):
    """
    Load equivalencias mapping from Text_Processing_Rules.xlsx.
    Returns dictionary mapping original terms to normalized terms.
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(text_processing_path):
        logger.warning(f"Text processing rules file not found at {text_processing_path}")
        return {}
    
    try:
        # Load equivalencias from Excel file
        df = pd.read_excel(text_processing_path, sheet_name='Equivalencias')
        equivalencias_map = {}
        
        for _, row in df.iterrows():
            original = str(row.get('Original', '')).strip().upper()
            normalized = str(row.get('Normalized', '')).strip().upper()
            if original and normalized:
                equivalencias_map[original] = normalized
        
        logger.info(f"Loaded {len(equivalencias_map)} equivalencias mappings")
        return equivalencias_map
        
    except Exception as e:
        logger.error(f"Error loading equivalencias map: {e}")
        return {}

# --- Main Processing Logic ---
def process_consolidado_record(record, equivalencias_map):
    """
    Process a single record from Consolidado.json.
    Returns processed record data or None if record should be skipped.
    """
    # Extract required fields only (removed unused columns)
    vin = record.get('vin_number')
    make = record.get('maker')
    year = record.get('fabrication_year')
    series = record.get('series')
    description = record.get('item_original_descripcion')
    sku = record.get('item_referencia')

    # Clean VIN if present (but don't discard record if invalid)
    cleaned_vin = clean_vin_for_training(vin) if vin else None

    # Clean and validate other fields
    make = str(make).strip() if make else None
    year = int(year) if year and str(year).isdigit() else None
    series = str(series).strip() if series else None
    description = str(description).strip() if description else None
    sku = str(sku).strip() if sku and str(sku).strip() else None

    # Determine record usefulness
    good_for_vin_training = (cleaned_vin and make and year and series)
    good_for_sku_training = (make and year and series and description and sku)

    if not (good_for_vin_training or good_for_sku_training):
        return None  # Skip - not useful for either training purpose

    # Process description if present
    normalized_descripcion = None
    equivalencia_row_id = None

    if description:
        try:
            normalized_descripcion = normalize_text(description, equivalencias_map)
            # For now, set equivalencia_row_id to None - could be enhanced later
            equivalencia_row_id = None
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error normalizing description '{description}': {e}")
            normalized_descripcion = description  # Use original if normalization fails

    return {
        'vin_number': cleaned_vin,
        'maker': make,
        'fabrication_year': year,
        'series': series,
        'original_descripcion': description,
        'normalized_descripcion': normalized_descripcion,
        'referencia': sku
    }

def process_consolidado_to_db(conn, consolidado_path, equivalencias_map):
    """
    Reads Consolidado.json, processes records, and inserts into the database.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing consolidado data from: {consolidado_path}")

    if not os.path.exists(consolidado_path):
        logger.error(f"Consolidado file not found at {consolidado_path}")
        return False

    # Statistics
    stats = {
        'total_records': 0,
        'total_items': 0,
        'inserted_records': 0,
        'skipped_insufficient_data': 0,
        'skipped_duplicates': 0,
        'vin_training_records': 0,
        'sku_training_records': 0,
        'both_training_records': 0
    }

    try:
        logger.info("Loading Consolidado.json...")
        with open(consolidado_path, 'r', encoding='utf-8') as f:
            all_records = json.load(f)

        stats['total_records'] = len(all_records)
        logger.info(f"Loaded {stats['total_records']} records from Consolidado.json")

        cursor = conn.cursor()

        # Process each record
        for record_idx, record in enumerate(all_records):
            if record_idx % 10000 == 0:
                logger.info(f"Processing record {record_idx + 1}/{stats['total_records']}")

            # Each record can have multiple items
            items = record.get('items', [])
            stats['total_items'] += len(items)

            # Extract VIN info from record level (simplified - removed unused fields)
            vin_number = record.get('vin_number')
            maker = record.get('maker')  # Field is 'maker'
            fabrication_year = record.get('fabrication_year')  # Field is 'fabrication_year'
            series = record.get('series')  # Field is 'series'

            # Process each item in the record
            for item in items:
                # Combine record-level and item-level data (simplified schema)
                combined_record = {
                    'vin_number': vin_number,
                    'maker': maker,
                    'fabrication_year': fabrication_year,
                    'series': series,
                    'item_original_descripcion': item.get('descripcion'),  # Field is 'descripcion'
                    'item_referencia': item.get('referencia')  # Field is 'referencia'
                }

                # Process the combined record
                processed_record = process_consolidado_record(combined_record, equivalencias_map)

                if processed_record is None:
                    stats['skipped_insufficient_data'] += 1
                    continue

                # Determine training usefulness for statistics
                has_vin_data = (processed_record['vin_number'] and
                               processed_record['maker'] and
                               processed_record['fabrication_year'] and
                               processed_record['series'])

                has_sku_data = (processed_record['maker'] and
                               processed_record['fabrication_year'] and
                               processed_record['series'] and
                               processed_record['normalized_descripcion'] and
                               processed_record['referencia'])

                if has_vin_data and has_sku_data:
                    stats['both_training_records'] += 1
                elif has_vin_data:
                    stats['vin_training_records'] += 1
                elif has_sku_data:
                    stats['sku_training_records'] += 1

                # Insert into database
                try:
                    cursor.execute('''
                    INSERT INTO processed_consolidado (
                        vin_number, maker, fabrication_year, series,
                        original_descripcion, normalized_descripcion, referencia
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        processed_record['vin_number'],
                        processed_record['maker'],
                        processed_record['fabrication_year'],
                        processed_record['series'],
                        processed_record['original_descripcion'],
                        processed_record['normalized_descripcion'],
                        processed_record['referencia']
                    ))
                    stats['inserted_records'] += 1

                except sqlite3.IntegrityError:
                    # Duplicate record
                    stats['skipped_duplicates'] += 1
                except Exception as e:
                    logger.error(f"Error inserting record: {e}")

        # Commit all changes
        conn.commit()

        # Log final statistics
        logger.info("=== Processing Complete ===")
        logger.info(f"Total records processed: {stats['total_records']}")
        logger.info(f"Total items processed: {stats['total_items']}")
        logger.info(f"Records inserted: {stats['inserted_records']}")
        logger.info(f"Records for VIN training only: {stats['vin_training_records']}")
        logger.info(f"Records for SKU training only: {stats['sku_training_records']}")
        logger.info(f"Records for both training: {stats['both_training_records']}")
        logger.info(f"Skipped (insufficient data): {stats['skipped_insufficient_data']}")
        logger.info(f"Skipped (duplicates): {stats['skipped_duplicates']}")

        return True

    except FileNotFoundError:
        logger.error(f"Consolidado file not found at {consolidado_path}")
        return False
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {consolidado_path}. File might be corrupted.")
        return False
    except MemoryError:
        logger.error(f"MemoryError loading {consolidado_path}. File may be too large.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        return False

# --- Main Execution ---
def main():
    """Main execution function."""
    logger = setup_logging()

    logger.info("=== Starting Unified Consolidado Processing ===")
    logger.info(f"Base path: {BASE_PATH}")
    logger.info(f"Source files directory: {SOURCE_FILES_DIR}")
    logger.info(f"Consolidado input: {CONSOLIDADO_PATH}")
    logger.info(f"Text processing rules: {TEXT_PROCESSING_PATH}")
    logger.info(f"Database output: {OUTPUT_DB_PATH}")

    # Check if input file exists
    if not os.path.exists(CONSOLIDADO_PATH):
        logger.error(f"Consolidado.json not found at {CONSOLIDADO_PATH}")
        logger.error("Please ensure Consolidado.json is in the Source_Files directory")
        return False

    # Load text processing rules
    logger.info("Loading text processing rules...")
    equivalencias_map = load_equivalencias_map(TEXT_PROCESSING_PATH)

    # Setup database
    logger.info("Setting up database...")
    conn = setup_database(OUTPUT_DB_PATH)
    if not conn:
        logger.error("Failed to setup database. Aborting.")
        return False

    try:
        # Process Consolidado.json
        logger.info("Starting Consolidado.json processing...")
        success = process_consolidado_to_db(conn, CONSOLIDADO_PATH, equivalencias_map)

        if success:
            # Final database statistics
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM processed_consolidado")
            total_records = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM processed_consolidado
                WHERE vin_number IS NOT NULL AND maker IS NOT NULL
                AND fabrication_year IS NOT NULL AND series IS NOT NULL
            """)
            vin_training_ready = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM processed_consolidado
                WHERE maker IS NOT NULL AND fabrication_year IS NOT NULL
                AND series IS NOT NULL AND normalized_descripcion IS NOT NULL
                AND sku IS NOT NULL
            """)
            sku_training_ready = cursor.fetchone()[0]

            logger.info("=== Final Database Statistics ===")
            logger.info(f"Total records in database: {total_records}")
            logger.info(f"Records ready for VIN training: {vin_training_ready}")
            logger.info(f"Records ready for SKU training: {sku_training_ready}")
            logger.info(f"Database created successfully: {OUTPUT_DB_PATH}")

        return success

    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

    logger.info("=== Unified Consolidado Processing Complete ===")

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

    print("\n" + "="*60)
    print("🎉 CONSOLIDADO PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Database created: processed_consolidado.db")
    print(f"📊 Ready for VIN and SKU training")
    print(f"📝 Check logs for detailed statistics")
    print("="*60)
