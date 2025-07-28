#!/usr/bin/env python3
"""
Script to verify the raw flattened consolidado database
Shows sample data and structure
"""

import sqlite3
import json
import pandas as pd

def verify_flattened_db():
    """Verify the raw flattened consolidado database"""
    
    db_file = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\TEMPORARY SCRIPTS TO REMOVE LATER\raw_consolidado_flattened_last_10k.db"
    
    print("="*80)
    print("RAW FLATTENED CONSOLIDADO DATABASE VERIFICATION")
    print("="*80)
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_file)
        
        # Get table info
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(consolidado_raw_flattened)")
        columns_info = cursor.fetchall()
        
        print(f"Database: {db_file}")
        print(f"Table: consolidado_raw_flattened")
        print(f"Columns: {len(columns_info)}")
        
        print("\nColumn Structure:")
        for col_info in columns_info:
            print(f"  {col_info[0]+1:2d}. {col_info[1]:<20} ({col_info[2]})")
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM consolidado_raw_flattened")
        row_count = cursor.fetchone()[0]
        print(f"\nTotal Rows: {row_count}")
        
        # Show sample data
        print("\n" + "="*80)
        print("SAMPLE DATA (First 5 rows)")
        print("="*80)
        
        df = pd.read_sql_query("SELECT * FROM consolidado_raw_flattened LIMIT 5", conn)
        
        for idx, row in df.iterrows():
            print(f"\n--- Row {idx + 1} ---")
            for col in df.columns:
                value = row[col]
                if col in ['_id', 'date']:
                    # These are JSON strings, try to parse and show nicely
                    try:
                        parsed = json.loads(value) if value else None
                        print(f"{col:<20}: {parsed}")
                    except:
                        print(f"{col:<20}: {value}")
                else:
                    # Show value as-is, truncate if too long
                    if isinstance(value, str) and len(str(value)) > 50:
                        print(f"{col:<20}: {str(value)[:50]}...")
                    else:
                        print(f"{col:<20}: {value}")
        
        # Show some statistics
        print("\n" + "="*80)
        print("DATA STATISTICS")
        print("="*80)
        
        # Count by maker
        cursor.execute("SELECT maker, COUNT(*) as count FROM consolidado_raw_flattened GROUP BY maker ORDER BY count DESC LIMIT 5")
        makers = cursor.fetchall()
        print("\nTop 5 Makers (by item count):")
        for maker, count in makers:
            print(f"  {maker:<15}: {count:>6} items")
        
        # Count by platform
        cursor.execute("SELECT platform, COUNT(*) as count FROM consolidado_raw_flattened GROUP BY platform ORDER BY count DESC")
        platforms = cursor.fetchall()
        print("\nPlatforms:")
        for platform, count in platforms:
            print(f"  {platform:<15}: {count:>6} items")
        
        # Sample descriptions
        cursor.execute("SELECT DISTINCT descripcion FROM consolidado_raw_flattened WHERE descripcion IS NOT NULL LIMIT 10")
        descriptions = cursor.fetchall()
        print("\nSample Descriptions:")
        for desc in descriptions:
            desc_text = desc[0][:60] + "..." if len(desc[0]) > 60 else desc[0]
            print(f"  - {desc_text}")
        
        # Count unique quotes
        cursor.execute("SELECT COUNT(DISTINCT quote) as unique_quotes FROM consolidado_raw_flattened")
        unique_quotes = cursor.fetchone()[0]
        print(f"\nUnique Quotes: {unique_quotes}")
        
        # Average items per quote
        avg_items = row_count / unique_quotes if unique_quotes > 0 else 0
        print(f"Average Items per Quote: {avg_items:.2f}")
        
        conn.close()
        
        print("\n✅ Database verification completed successfully!")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_flattened_db()
