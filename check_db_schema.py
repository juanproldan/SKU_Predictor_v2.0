#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check Database Schema

Quick script to check the database schema and find VIN column name.
"""

import os
import sys
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_schema():
    """Check database schema."""
    
    try:
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("PRAGMA table_info(processed_consolidado)")
        columns = cursor.fetchall()
        
        print("📋 Database Schema:")
        for col in columns:
            print(f"   {col[1]} ({col[2]})")
        
        # Check for VIN-like columns
        print("\n🔍 Looking for VIN data...")
        for col in columns:
            col_name = col[1]
            if 'vin' in col_name.lower():
                cursor.execute(f"SELECT COUNT(*) FROM processed_consolidado WHERE {col_name} IS NOT NULL AND {col_name} != ''")
                count = cursor.fetchone()[0]
                print(f"   {col_name}: {count:,} non-empty values")

                if count > 0:
                    # Check for real VINs (not all zeros)
                    cursor.execute(f"SELECT COUNT(*) FROM processed_consolidado WHERE {col_name} IS NOT NULL AND {col_name} != '' AND {col_name} != '00000000000000000'")
                    real_count = cursor.fetchone()[0]
                    print(f"   Non-zero VINs: {real_count:,}")

                    if real_count > 0:
                        cursor.execute(f"SELECT {col_name} FROM processed_consolidado WHERE {col_name} IS NOT NULL AND {col_name} != '' AND {col_name} != '00000000000000000' LIMIT 10")
                        samples = cursor.fetchall()
                        print(f"   Real VIN samples: {[s[0] for s in samples]}")

                        # Check VIN length distribution
                        cursor.execute(f"SELECT LENGTH({col_name}) as len, COUNT(*) as count FROM processed_consolidado WHERE {col_name} IS NOT NULL AND {col_name} != '' AND {col_name} != '00000000000000000' GROUP BY LENGTH({col_name}) ORDER BY count DESC LIMIT 5")
                        lengths = cursor.fetchall()
                        print(f"   VIN lengths: {lengths}")
                    else:
                        cursor.execute(f"SELECT {col_name} FROM processed_consolidado WHERE {col_name} IS NOT NULL AND {col_name} != '' LIMIT 5")
                        samples = cursor.fetchall()
                        print(f"   All samples: {[s[0] for s in samples]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_schema()
