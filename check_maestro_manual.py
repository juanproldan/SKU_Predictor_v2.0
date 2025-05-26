#!/usr/bin/env python3
"""
Quick script to check for MANUAL entries in Maestro.xlsx
"""

import pandas as pd
import os

def check_maestro_for_manual():
    maestro_path = os.path.join("data", "Maestro.xlsx")
    
    if not os.path.exists(maestro_path):
        print(f"❌ Maestro.xlsx not found at {maestro_path}")
        return
    
    try:
        print(f"📂 Loading Maestro.xlsx from {maestro_path}")
        df = pd.read_excel(maestro_path)
        
        print(f"📊 Total records in Maestro: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        
        if 'Confirmed_SKU' in df.columns:
            # Check for MANUAL entries
            manual_entries = df[df['Confirmed_SKU'].str.upper() == 'MANUAL']
            
            if len(manual_entries) > 0:
                print(f"\n🚨 Found {len(manual_entries)} MANUAL entries in Maestro.xlsx:")
                print("=" * 60)
                
                for idx, row in manual_entries.iterrows():
                    print(f"Row {idx + 2}:")  # +2 because Excel is 1-indexed and has header
                    print(f"  Maestro_ID: {row.get('Maestro_ID', 'N/A')}")
                    print(f"  VIN_Make: {row.get('VIN_Make', 'N/A')}")
                    print(f"  VIN_Year_Min: {row.get('VIN_Year_Min', 'N/A')}")
                    print(f"  VIN_Series_Trim: {row.get('VIN_Series_Trim', 'N/A')}")
                    print(f"  Original_Description_Input: {row.get('Original_Description_Input', 'N/A')}")
                    print(f"  Normalized_Description_Input: {row.get('Normalized_Description_Input', 'N/A')}")
                    print(f"  Confirmed_SKU: {row.get('Confirmed_SKU', 'N/A')}")
                    print(f"  Source: {row.get('Source', 'N/A')}")
                    print(f"  Date_Added: {row.get('Date_Added', 'N/A')}")
                    print("-" * 40)
                
                print(f"\n💡 These entries should be cleaned up or have proper SKUs assigned.")
                
            else:
                print(f"\n✅ No MANUAL entries found in Confirmed_SKU column")
                
            # Also check for other invalid SKUs
            invalid_skus = {'UNKNOWN', 'N/A', 'NULL', 'NONE', 'TBD', 'PENDING', 'MANUAL'}
            
            print(f"\n🔍 Checking for other invalid SKUs...")
            for invalid_sku in invalid_skus:
                invalid_entries = df[df['Confirmed_SKU'].str.upper() == invalid_sku]
                if len(invalid_entries) > 0:
                    print(f"  ⚠️  Found {len(invalid_entries)} entries with SKU '{invalid_sku}'")
                else:
                    print(f"  ✅ No entries with SKU '{invalid_sku}'")
                    
            # Show sample of actual SKUs
            print(f"\n📋 Sample of actual SKUs in Maestro:")
            valid_skus = df[~df['Confirmed_SKU'].str.upper().isin(invalid_skus)]['Confirmed_SKU'].dropna().unique()
            for i, sku in enumerate(valid_skus[:10]):  # Show first 10
                print(f"  {i+1}. {sku}")
            if len(valid_skus) > 10:
                print(f"  ... and {len(valid_skus) - 10} more")
                
        else:
            print(f"❌ 'Confirmed_SKU' column not found in Maestro.xlsx")
            
    except Exception as e:
        print(f"❌ Error reading Maestro.xlsx: {e}")

if __name__ == "__main__":
    print("🔍 Checking Maestro.xlsx for MANUAL entries...")
    print("=" * 50)
    check_maestro_for_manual()
