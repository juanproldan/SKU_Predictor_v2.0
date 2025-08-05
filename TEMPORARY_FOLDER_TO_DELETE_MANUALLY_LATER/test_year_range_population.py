#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to populate year range tables
"""

import sqlite3
import os
import sys
from collections import defaultdict

# Add src to path
sys.path.append('src')

def get_base_path():
    """Get the base path for the application."""
    return os.path.join(os.getcwd(), "Fixacar_SKU_Predictor_CLIENT")

def detect_year_ranges(years):
    """
    Detect year ranges from a list of years, allowing gaps of 1-2 years.
    Returns a list of (start_year, end_year) tuples.
    """
    if not years:
        return []
    
    years = sorted(set(years))  # Remove duplicates and sort
    
    if len(years) == 1:
        return [(years[0], years[0])]
    
    # For automotive parts, create one range from min to max year
    # This reflects the reality that parts work in year ranges
    return [(min(years), max(years))]

def populate_year_range_tables():
    """Populate the year range tables from existing data."""
    
    db_path = os.path.join(get_base_path(), "Source_Files", "processed_consolidado.db")
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return False
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if main table exists and has data
        cursor.execute("SELECT COUNT(*) FROM processed_consolidado")
        total_records = cursor.fetchone()[0]
        print(f"Total records in processed_consolidado: {total_records:,}")
        
        if total_records == 0:
            print("No data in processed_consolidado table")
            return False
        
        # Clear existing year range data
        cursor.execute("DELETE FROM sku_year_ranges")
        cursor.execute("DELETE FROM vin_year_ranges")
        print("Cleared existing year range data")
        
        # Aggregate SKU data by maker, series, description, referencia
        print("Aggregating SKU data...")
        cursor.execute("""
            SELECT maker, series, descripcion, normalized_descripcion, referencia, 
                   GROUP_CONCAT(model) as years, COUNT(*) as frequency
            FROM processed_consolidado 
            WHERE referencia IS NOT NULL AND referencia != '' AND referencia != 'UNKNOWN'
            GROUP BY maker, series, descripcion, referencia
        """)
        
        sku_data = cursor.fetchall()
        print(f"Found {len(sku_data):,} unique SKU combinations")
        
        # Process SKU year ranges
        sku_ranges_inserted = 0
        for row in sku_data:
            maker, series, descripcion, normalized_descripcion, referencia, years_str, frequency = row
            
            # Parse years
            years = []
            if years_str:
                for year_str in years_str.split(','):
                    try:
                        years.append(int(year_str))
                    except ValueError:
                        continue
            
            if not years:
                continue
            
            # Detect year ranges
            year_ranges = detect_year_ranges(years)
            
            for start_year, end_year in year_ranges:
                cursor.execute("""
                    INSERT OR REPLACE INTO sku_year_ranges 
                    (maker, series, descripcion, normalized_descripcion, referencia, start_year, end_year, frequency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (maker, series, descripcion, normalized_descripcion, referencia, start_year, end_year, frequency))
                sku_ranges_inserted += 1
        
        print(f"Inserted {sku_ranges_inserted:,} SKU year ranges")
        
        # Aggregate VIN data by maker, series
        print("Aggregating VIN data...")
        cursor.execute("""
            SELECT maker, series, GROUP_CONCAT(model) as years, COUNT(*) as frequency
            FROM processed_consolidado 
            WHERE maker IS NOT NULL AND series IS NOT NULL
            GROUP BY maker, series
        """)
        
        vin_data = cursor.fetchall()
        print(f"Found {len(vin_data):,} unique VIN combinations")
        
        # Process VIN year ranges
        vin_ranges_inserted = 0
        for row in vin_data:
            maker, series, years_str, frequency = row
            
            # Parse years
            years = []
            if years_str:
                for year_str in years_str.split(','):
                    try:
                        years.append(int(year_str))
                    except ValueError:
                        continue
            
            if not years:
                continue
            
            # Detect year ranges
            year_ranges = detect_year_ranges(years)
            
            for start_year, end_year in year_ranges:
                cursor.execute("""
                    INSERT OR REPLACE INTO vin_year_ranges 
                    (maker, series, start_year, end_year, frequency)
                    VALUES (?, ?, ?, ?, ?)
                """, (maker, series, start_year, end_year, frequency))
                vin_ranges_inserted += 1
        
        print(f"Inserted {vin_ranges_inserted:,} VIN year ranges")
        
        # Commit changes
        conn.commit()
        
        # Verify results
        cursor.execute("SELECT COUNT(*) FROM sku_year_ranges")
        sku_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM vin_year_ranges")
        vin_count = cursor.fetchone()[0]
        
        print(f"\n=== Year Range Population Complete ===")
        print(f"SKU year ranges: {sku_count:,}")
        print(f"VIN year ranges: {vin_count:,}")
        
        # Show some sample data
        print(f"\n=== Sample SKU Year Ranges ===")
        cursor.execute("""
            SELECT maker, series, referencia, start_year, end_year, frequency 
            FROM sku_year_ranges 
            ORDER BY frequency DESC 
            LIMIT 5
        """)
        
        for row in cursor.fetchall():
            maker, series, referencia, start_year, end_year, frequency = row
            print(f"  {maker}/{series} - {referencia}: {start_year}-{end_year} (freq: {frequency})")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    success = populate_year_range_tables()
    if success:
        print("✅ Year range tables populated successfully")
    else:
        print("❌ Failed to populate year range tables")
