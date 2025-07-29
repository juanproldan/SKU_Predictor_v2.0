#!/usr/bin/env python3
"""
Fix Database Schema - Update processed_consolidado.db to use correct field names
This script will recreate the database with the correct schema using original consolidado.json field names.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from unified_consolidado_processor import main as process_consolidado

def main():
    """Main function to fix the database schema"""
    print("="*80)
    print("FIXING DATABASE SCHEMA")
    print("="*80)
    print("Recreating processed_consolidado.db with correct field names:")
    print("  - maker")
    print("  - model") 
    print("  - series")
    print("  - descripcion")
    print("  - referencia")
    print("="*80)
    
    # Paths
    db_path = project_root / "Source_Files" / "processed_consolidado.db"
    consolidado_path = project_root / "Source_Files" / "Consolidado.json"
    
    print(f"Database path: {db_path}")
    print(f"Consolidado path: {consolidado_path}")
    
    # Check if files exist
    if not consolidado_path.exists():
        print(f"❌ Error: Consolidado.json not found at {consolidado_path}")
        return False
    
    # Backup existing database if it exists
    if db_path.exists():
        backup_path = db_path.with_suffix('.db.backup')
        print(f"📁 Backing up existing database to {backup_path}")
        import shutil
        shutil.copy2(db_path, backup_path)
        
        # Remove old database
        print(f"🗑️  Removing old database")
        os.remove(db_path)
    
    # Process the consolidado file with the new schema
    print(f"🔄 Processing Consolidado.json with new schema...")
    try:
        success = process_consolidado()
        
        if success:
            print("✅ Database recreated successfully with correct schema!")
            
            # Verify the new schema
            print("\n📊 Verifying new database schema...")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("PRAGMA table_info(processed_consolidado)")
            columns = cursor.fetchall()
            
            print("Database columns:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
            
            # Get record count
            cursor.execute("SELECT COUNT(*) FROM processed_consolidado")
            count = cursor.fetchone()[0]
            print(f"\nTotal records: {count}")
            
            # Sample a few records
            cursor.execute("SELECT * FROM processed_consolidado LIMIT 3")
            samples = cursor.fetchall()
            
            print("\nSample records:")
            for i, record in enumerate(samples, 1):
                print(f"  Record {i}: {record}")
            
            conn.close()
            return True
            
        else:
            print("❌ Failed to process consolidado file")
            return False
            
    except Exception as e:
        print(f"❌ Error processing consolidado file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Database schema fix completed successfully!")
        print("You can now run the training scripts and main application.")
    else:
        print("\n💥 Database schema fix failed!")
        print("Please check the error messages above.")
