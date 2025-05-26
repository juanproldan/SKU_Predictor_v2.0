#!/usr/bin/env python3
"""
Script to clean up UNKNOWN and invalid SKU entries from Maestro.xlsx
This improves data quality by removing entries that don't contribute to learning
"""

import pandas as pd
import os
from datetime import datetime

def clean_maestro_unknown():
    """Remove UNKNOWN and invalid SKU entries from Maestro.xlsx"""
    
    maestro_path = os.path.join("data", "Maestro.xlsx")
    
    if not os.path.exists(maestro_path):
        print(f"❌ Maestro.xlsx not found at {maestro_path}")
        return
    
    # Create backup first
    backup_path = os.path.join("data", f"Maestro_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    try:
        print(f"🔄 Loading Maestro.xlsx...")
        df = pd.read_excel(maestro_path)
        
        print(f"📊 Original records: {len(df)}")
        
        # Create backup
        df.to_excel(backup_path, index=False)
        print(f"💾 Backup created: {backup_path}")
        
        # Define invalid SKUs to remove
        invalid_skus = {'UNKNOWN', 'N/A', 'NULL', 'NONE', '', 'TBD', 'PENDING', 'MANUAL'}
        
        # Show what will be removed
        print(f"\n🔍 Analyzing entries to remove...")
        for invalid_sku in invalid_skus:
            if 'Confirmed_SKU' in df.columns:
                invalid_entries = df[df['Confirmed_SKU'].str.upper() == invalid_sku]
                if len(invalid_entries) > 0:
                    print(f"  📋 Found {len(invalid_entries)} entries with SKU '{invalid_sku}':")
                    for idx, row in invalid_entries.head(3).iterrows():  # Show first 3
                        desc = row.get('Original_Description_Input', 'N/A')
                        print(f"    - {desc[:50]}...")
                    if len(invalid_entries) > 3:
                        print(f"    ... and {len(invalid_entries) - 3} more")
        
        # Filter out invalid SKUs
        if 'Confirmed_SKU' in df.columns:
            # Keep only entries with valid SKUs
            original_count = len(df)
            df_clean = df[~df['Confirmed_SKU'].str.upper().isin(invalid_skus)]
            removed_count = original_count - len(df_clean)
            
            print(f"\n🧹 Cleaning results:")
            print(f"  Original entries: {original_count}")
            print(f"  Entries removed: {removed_count}")
            print(f"  Clean entries: {len(df_clean)}")
            
            if removed_count > 0:
                # Save cleaned data
                df_clean.to_excel(maestro_path, index=False)
                print(f"✅ Cleaned Maestro.xlsx saved!")
                
                # Show sample of remaining valid SKUs
                print(f"\n📋 Sample of remaining valid SKUs:")
                valid_skus = df_clean['Confirmed_SKU'].dropna().unique()
                for i, sku in enumerate(valid_skus[:10]):  # Show first 10
                    print(f"  {i+1}. {sku}")
                if len(valid_skus) > 10:
                    print(f"  ... and {len(valid_skus) - 10} more")
                    
                print(f"\n🎯 Data quality improved! Only valid SKUs remain in Maestro.")
                print(f"   This will improve the learning system's effectiveness.")
                
            else:
                print(f"✅ No invalid entries found - Maestro is already clean!")
                # Remove backup since no changes were made
                os.remove(backup_path)
                print(f"🗑️  Backup removed (no changes needed)")
                
        else:
            print(f"❌ 'Confirmed_SKU' column not found in Maestro.xlsx")
            
    except Exception as e:
        print(f"❌ Error processing Maestro.xlsx: {e}")
        if os.path.exists(backup_path):
            print(f"💾 Backup preserved at: {backup_path}")

if __name__ == "__main__":
    print("🧹 Cleaning UNKNOWN entries from Maestro.xlsx...")
    print("=" * 60)
    clean_maestro_unknown()
    print("=" * 60)
    print("✅ Cleanup complete!")
