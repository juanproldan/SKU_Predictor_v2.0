#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test database setup functionality
"""

import sqlite3
import os
import sys
import logging

# Add src to path
sys.path.append('src')

def get_base_path():
    """Get the base path for the application."""
    return os.path.join(os.getcwd(), "Fixacar_SKU_Predictor_CLIENT")

def test_database_setup():
    """Test the database setup functionality."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    db_path = os.path.join(get_base_path(), "Source_Files", "processed_consolidado.db")
    
    print(f"Testing database setup at: {db_path}")
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Removed existing database")
    
    try:
        # Create new database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create main table
        cursor.execute('''
        CREATE TABLE processed_consolidado (
            vin_number TEXT,                    -- For VIN training
            maker TEXT,                         -- For both VIN & SKU training
            model INTEGER,                      -- For both VIN & SKU training
            series TEXT,                        -- For both VIN & SKU training
            descripcion TEXT,                   -- Original description from consolidado.json (may be NULL)
            normalized_descripcion TEXT,        -- Normalized description for SKU training (may be NULL)
            referencia TEXT,                    -- For SKU training (may be NULL)
            UNIQUE(vin_number, descripcion, referencia) -- Prevent duplicates
        )
        ''')
        print("✅ Main table created")
        
        # Drop existing year range tables if they exist (for clean rebuild)
        cursor.execute('DROP TABLE IF EXISTS sku_year_ranges')
        cursor.execute('DROP TABLE IF EXISTS vin_year_ranges')
        
        # Create aggregated year range tables for improved frequency counting
        cursor.execute('''
        CREATE TABLE sku_year_ranges (
            maker TEXT,
            series TEXT,
            descripcion TEXT,
            normalized_descripcion TEXT,
            referencia TEXT,
            start_year INTEGER,
            end_year INTEGER,
            frequency INTEGER,
            PRIMARY KEY (maker, series, descripcion, referencia)
        )
        ''')
        print("✅ SKU year ranges table created")

        cursor.execute('''
        CREATE TABLE vin_year_ranges (
            maker TEXT,
            series TEXT,
            start_year INTEGER,
            end_year INTEGER,
            frequency INTEGER,
            PRIMARY KEY (maker, series)
        )
        ''')
        print("✅ VIN year ranges table created")

        # Create indexes for better query performance
        cursor.execute('CREATE INDEX idx_vin_training ON processed_consolidado (vin_number, maker, model, series)')
        cursor.execute('CREATE INDEX idx_referencia_training ON processed_consolidado (maker, model, series, referencia)')
        cursor.execute('CREATE INDEX idx_exact_match ON processed_consolidado (maker, model, series, normalized_descripcion)')
        cursor.execute('CREATE INDEX idx_description_search ON processed_consolidado (normalized_descripcion)')
        print("✅ Main table indexes created")

        # Create indexes for year range tables
        cursor.execute('CREATE INDEX idx_sku_year_range_lookup ON sku_year_ranges (maker, series, start_year, end_year)')
        cursor.execute('CREATE INDEX idx_sku_frequency ON sku_year_ranges (frequency DESC)')
        cursor.execute('CREATE INDEX idx_vin_year_range_lookup ON vin_year_ranges (maker, series, start_year, end_year)')
        cursor.execute('CREATE INDEX idx_vin_frequency ON vin_year_ranges (frequency DESC)')
        print("✅ Year range indexes created")

        conn.commit()
        print("✅ Database setup completed successfully")
        
        # Test inserting a sample record
        cursor.execute("""
            INSERT INTO processed_consolidado 
            (vin_number, maker, model, series, descripcion, normalized_descripcion, referencia)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("TEST123", "Toyota", 2020, "Corolla", "test part", "test part normalized", "TEST-REF"))
        
        conn.commit()
        print("✅ Sample record inserted")
        
        # Verify the record
        cursor.execute("SELECT COUNT(*) FROM processed_consolidado")
        count = cursor.fetchone()[0]
        print(f"✅ Records in database: {count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_database_setup()
    if success:
        print("\n🎉 Database setup test passed!")
    else:
        print("\n💥 Database setup test failed!")
