#!/usr/bin/env python3
"""
Script to create a raw database from consolidado.json
- Takes last 10,000 entries only
- No data processing or normalization
- Keeps all original columns and names
- No data modifications whatsoever
"""

import json
import sqlite3
import pandas as pd
import os
from datetime import datetime

def create_raw_consolidado_db():
    """Create raw database from consolidado.json with last 10,000 entries"""
    
    # Define paths
    source_file = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\Source_Files\consolidado.json"
    output_file = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\TEMPORARY SCRIPTS TO REMOVE LATER\raw_consolidado_last_10k.db"
    
    print(f"Starting raw consolidado database creation...")
    print(f"Source: {source_file}")
    print(f"Output: {output_file}")
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"ERROR: Source file not found: {source_file}")
        return False
    
    try:
        # Load JSON data
        print("Loading JSON data...")
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total entries in JSON: {len(data)}")
        
        # Take only the last 10,000 entries
        if len(data) > 10000:
            data = data[-10000:]
            print(f"Taking last 10,000 entries")
        else:
            print(f"File has {len(data)} entries (less than 10,000), using all")
        
        # Convert to DataFrame (this preserves all original data and column names)
        print("Converting to DataFrame...")
        df = pd.DataFrame(data)

        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Handle complex data types (convert dict/list columns to JSON strings)
        print("Processing complex data types...")
        for column in df.columns:
            # Check if column contains complex data types (dict, list)
            sample_value = df[column].iloc[0] if len(df) > 0 else None
            if isinstance(sample_value, (dict, list)):
                print(f"  Converting column '{column}' (complex type) to JSON string")
                df[column] = df[column].apply(lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None)

        # Create SQLite database
        print("Creating SQLite database...")

        # Remove existing file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
            print("Removed existing database file")

        # Create database connection
        conn = sqlite3.connect(output_file)

        # Save DataFrame to SQLite (preserves all original data)
        # if_exists='replace' ensures we create a fresh table
        # index=False prevents adding an extra index column
        df.to_sql('consolidado_raw', conn, if_exists='replace', index=False)
        
        # Get table info
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(consolidado_raw)")
        columns_info = cursor.fetchall()
        
        print(f"\nDatabase created successfully!")
        print(f"Table: consolidado_raw")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(columns_info)}")
        
        print("\nColumn information:")
        for col_info in columns_info:
            print(f"  - {col_info[1]} ({col_info[2]})")
        
        # Close connection
        conn.close()
        
        print(f"\nRaw database saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("RAW CONSOLIDADO DATABASE CREATOR")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = create_raw_consolidado_db()
    
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if success:
        print("✅ SUCCESS: Raw database created successfully!")
    else:
        print("❌ FAILED: Database creation failed!")
