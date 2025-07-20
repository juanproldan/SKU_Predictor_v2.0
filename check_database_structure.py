#!/usr/bin/env python3
"""
Check database structure to understand table names and schema.
"""

import sqlite3
import pandas as pd

def check_database_structure():
    """Check what tables and columns exist in the database."""
    
    db_path = 'data/consolidado.db'
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get all table names
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(tables_query, conn)
        
        print("=== DATABASE TABLES ===")
        for table in tables['name']:
            print(f"📋 Table: {table}")
            
            # Get table schema
            schema_query = f"PRAGMA table_info({table});"
            schema = pd.read_sql_query(schema_query, conn)
            
            print("   Columns:")
            for _, col in schema.iterrows():
                print(f"      {col['name']} ({col['type']})")
            
            # Get sample data
            sample_query = f"SELECT * FROM {table} LIMIT 3;"
            try:
                sample = pd.read_sql_query(sample_query, conn)
                print(f"   Sample data ({len(sample)} rows):")
                for _, row in sample.iterrows():
                    print(f"      {dict(row)}")
            except Exception as e:
                print(f"   Error getting sample data: {e}")
            
            print("-" * 50)
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_database_structure()
