#!/usr/bin/env python3
"""
Script to create a raw flattened database from consolidado.json
- Takes last 10,000 entries only
- Flattens items array so each item becomes a separate row
- No data processing or normalization
- Keeps all original columns and names
- No data modifications whatsoever
"""

import json
import sqlite3
import pandas as pd
import os
from datetime import datetime

def create_raw_flattened_consolidado_db():
    """Create raw flattened database from consolidado.json with last 10,000 entries"""
    
    # Define paths
    source_file = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\Source_Files\consolidado.json"
    output_file = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\TEMPORARY SCRIPTS TO REMOVE LATER\raw_consolidado_flattened_last_10k.db"
    
    print(f"Starting raw flattened consolidado database creation...")
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
        
        # Flatten the data - each item becomes a separate row
        print("Flattening items into separate rows...")
        flattened_data = []
        
        for record in data:
            # Get the base record data (everything except items)
            base_record = {k: v for k, v in record.items() if k != 'items'}
            
            # Convert complex fields to JSON strings for base record
            for key, value in base_record.items():
                if isinstance(value, (dict, list)):
                    base_record[key] = json.dumps(value, ensure_ascii=False)
            
            # Get items array
            items = record.get('items', [])
            
            if items and isinstance(items, list):
                # Create a row for each item
                for item in items:
                    # Start with base record
                    flattened_record = base_record.copy()
                    
                    # Add item fields directly as columns
                    if isinstance(item, dict):
                        for item_key, item_value in item.items():
                            # Convert complex item values to JSON strings if needed
                            if isinstance(item_value, (dict, list)):
                                flattened_record[item_key] = json.dumps(item_value, ensure_ascii=False)
                            else:
                                flattened_record[item_key] = item_value
                    
                    flattened_data.append(flattened_record)
            else:
                # If no items, still add the base record
                flattened_data.append(base_record)
        
        print(f"Flattened to {len(flattened_data)} rows")
        
        # Convert to DataFrame
        print("Converting to DataFrame...")
        df = pd.DataFrame(flattened_data)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Create SQLite database
        print("Creating SQLite database...")
        
        # Remove existing file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
            print("Removed existing database file")
        
        # Create database connection
        conn = sqlite3.connect(output_file)
        
        # Save DataFrame to SQLite (preserves all original data)
        df.to_sql('consolidado_raw_flattened', conn, if_exists='replace', index=False)
        
        # Get table info
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(consolidado_raw_flattened)")
        columns_info = cursor.fetchall()
        
        print(f"\nDatabase created successfully!")
        print(f"Table: consolidado_raw_flattened")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(columns_info)}")
        
        print("\nColumn information:")
        for col_info in columns_info:
            print(f"  - {col_info[1]} ({col_info[2]})")
        
        # Close connection
        conn.close()
        
        print(f"\nRaw flattened database saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("RAW FLATTENED CONSOLIDADO DATABASE CREATOR")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = create_raw_flattened_consolidado_db()
    
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if success:
        print("✅ SUCCESS: Raw flattened database created successfully!")
    else:
        print("❌ FAILED: Database creation failed!")
