#!/usr/bin/env python3
"""
Script to verify the raw consolidado database
Shows sample data and structure
"""

import sqlite3
import json
import pandas as pd

def verify_raw_db():
    """Verify the raw consolidado database"""
    
    db_file = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\TEMPORARY SCRIPTS TO REMOVE LATER\raw_consolidado_last_10k.db"
    
    print("="*60)
    print("RAW CONSOLIDADO DATABASE VERIFICATION")
    print("="*60)
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_file)
        
        # Get table info
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(consolidado_raw)")
        columns_info = cursor.fetchall()
        
        print(f"Database: {db_file}")
        print(f"Table: consolidado_raw")
        print(f"Columns: {len(columns_info)}")
        
        print("\nColumn Structure:")
        for col_info in columns_info:
            print(f"  {col_info[0]+1:2d}. {col_info[1]:<20} ({col_info[2]})")
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM consolidado_raw")
        row_count = cursor.fetchone()[0]
        print(f"\nTotal Rows: {row_count}")
        
        # Show sample data
        print("\n" + "="*60)
        print("SAMPLE DATA (First 3 rows)")
        print("="*60)
        
        df = pd.read_sql_query("SELECT * FROM consolidado_raw LIMIT 3", conn)
        
        for idx, row in df.iterrows():
            print(f"\n--- Row {idx + 1} ---")
            for col in df.columns:
                value = row[col]
                if col in ['_id', 'date', 'items']:
                    # These are JSON strings, try to parse and show nicely
                    try:
                        parsed = json.loads(value) if value else None
                        if col == 'items' and parsed:
                            print(f"{col:<20}: {len(parsed)} items")
                            if len(parsed) > 0:
                                first_item = parsed[0]
                                print(f"{'':20}  First item: {first_item.get('description', 'N/A')[:50]}...")
                        else:
                            print(f"{col:<20}: {parsed}")
                    except:
                        print(f"{col:<20}: {value}")
                else:
                    print(f"{col:<20}: {value}")
        
        # Show some statistics
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        
        # Count by maker
        cursor.execute("SELECT maker, COUNT(*) as count FROM consolidado_raw GROUP BY maker ORDER BY count DESC LIMIT 5")
        makers = cursor.fetchall()
        print("\nTop 5 Makers:")
        for maker, count in makers:
            print(f"  {maker:<15}: {count:>5} records")
        
        # Count by year
        cursor.execute("SELECT fabrication_year, COUNT(*) as count FROM consolidado_raw GROUP BY fabrication_year ORDER BY count DESC LIMIT 5")
        years = cursor.fetchall()
        print("\nTop 5 Years:")
        for year, count in years:
            print(f"  {year:<15}: {count:>5} records")
        
        conn.close()
        
        print("\n✅ Database verification completed successfully!")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    verify_raw_db()
