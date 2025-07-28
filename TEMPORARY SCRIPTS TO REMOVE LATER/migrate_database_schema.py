#!/usr/bin/env python3
"""
Script to migrate the existing processed_consolidado.db to use the new field names.
This will rename columns to match the original consolidado.json structure.
"""

import sqlite3
import os
from pathlib import Path

def migrate_database_schema():
    """Migrate the database schema to use new field names"""
    
    db_path = "Source_Files/processed_consolidado.db"
    backup_path = "Source_Files/processed_consolidado_backup.db"
    
    print("="*80)
    print("DATABASE SCHEMA MIGRATION")
    print("="*80)
    print("Migrating processed_consolidado.db to use original field names:")
    print("  vin_make -> maker")
    print("  vin_year -> model") 
    print("  vin_series -> series")
    print("  original_description -> original_descripcion")
    print("  normalized_description -> normalized_descripcion")
    print("  sku -> referencia")
    print("="*80)
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        # Create backup
        print(f"📦 Creating backup: {backup_path}")
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created successfully")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check current schema
        cursor.execute("PRAGMA table_info(processed_consolidado)")
        current_columns = cursor.fetchall()
        print(f"\n📋 Current schema ({len(current_columns)} columns):")
        for col in current_columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Create new table with correct schema
        print(f"\n🔄 Creating new table with updated schema...")
        cursor.execute('''
        CREATE TABLE processed_consolidado_new (
            vin_number TEXT,
            maker TEXT,
            model INTEGER,
            series TEXT,
            original_descripcion TEXT,
            normalized_descripcion TEXT,
            referencia TEXT,
            UNIQUE(vin_number, original_descripcion, referencia)
        )
        ''')
        
        # Copy data with column mapping
        print(f"📊 Copying data with column mapping...")
        cursor.execute('''
        INSERT INTO processed_consolidado_new (
            vin_number, maker, model, series, 
            original_descripcion, normalized_descripcion, referencia
        )
        SELECT 
            vin_number, vin_make, vin_year, vin_series,
            original_description, normalized_description, sku
        FROM processed_consolidado
        ''')
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM processed_consolidado_new")
        row_count = cursor.fetchone()[0]
        print(f"✅ Copied {row_count} rows successfully")
        
        # Drop old table and rename new one
        print(f"🔄 Replacing old table...")
        cursor.execute("DROP TABLE processed_consolidado")
        cursor.execute("ALTER TABLE processed_consolidado_new RENAME TO processed_consolidado")
        
        # Create indexes for better performance
        print(f"🔧 Creating optimized indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_vin_training ON processed_consolidado (vin_number, maker, model, series)",
            "CREATE INDEX IF NOT EXISTS idx_sku_training ON processed_consolidado (maker, model, series, referencia)",
            "CREATE INDEX IF NOT EXISTS idx_exact_match ON processed_consolidado (maker, model, series, normalized_descripcion)",
            "CREATE INDEX IF NOT EXISTS idx_sku_frequency ON processed_consolidado (referencia)",
            "CREATE INDEX IF NOT EXISTS idx_description_search ON processed_consolidado (normalized_descripcion)",
            "CREATE INDEX IF NOT EXISTS idx_vin_lookup ON processed_consolidado (vin_number)",
            "CREATE INDEX IF NOT EXISTS idx_make_year ON processed_consolidado (maker, model)"
        ]
        
        for idx_sql in indexes:
            cursor.execute(idx_sql)
            print(f"  ✅ Created index")
        
        # Commit changes
        conn.commit()
        
        # Verify new schema
        cursor.execute("PRAGMA table_info(processed_consolidado)")
        new_columns = cursor.fetchall()
        print(f"\n✅ New schema ({len(new_columns)} columns):")
        for col in new_columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Verify data integrity
        cursor.execute("SELECT COUNT(*) FROM processed_consolidado")
        final_count = cursor.fetchone()[0]
        print(f"\n📊 Final verification:")
        print(f"  - Total rows: {final_count}")
        print(f"  - Data integrity: {'✅ PASSED' if final_count == row_count else '❌ FAILED'}")
        
        conn.close()
        
        print(f"\n🎉 Database migration completed successfully!")
        print(f"📦 Backup available at: {backup_path}")
        print(f"🗃️  Updated database: {db_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Restore backup if it exists
        if os.path.exists(backup_path):
            print(f"🔄 Restoring backup...")
            shutil.copy2(backup_path, db_path)
            print(f"✅ Backup restored")
        
        return False

def main():
    """Main function"""
    success = migrate_database_schema()
    
    if success:
        print(f"\n🚀 Next steps:")
        print(f"1. Test the main application")
        print(f"2. Verify SKU predictions work correctly")
        print(f"3. Remove backup file when satisfied")
    else:
        print(f"\n⚠️  Migration failed - please check the errors above")
    
    return success

if __name__ == "__main__":
    main()
