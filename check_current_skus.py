#!/usr/bin/env python3
"""
Check current SKUs in Maestro.xlsx to see what's actually saved
"""

import pandas as pd
import os

def check_current_skus():
    """Check what SKUs are currently in Maestro.xlsx"""
    
    maestro_path = os.path.join("data", "Maestro.xlsx")
    
    if not os.path.exists(maestro_path):
        print(f"❌ Maestro.xlsx not found at {maestro_path}")
        return
    
    try:
        print(f"📂 Loading Maestro.xlsx...")
        df = pd.read_excel(maestro_path)
        
        print(f"📊 Total records: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        
        if 'Confirmed_SKU' in df.columns:
            print(f"\n🔍 Unique SKUs in Maestro:")
            sku_counts = df['Confirmed_SKU'].value_counts()
            
            for sku, count in sku_counts.items():
                print(f"  {sku}: {count} times")
            
            print(f"\n📈 SKU Statistics:")
            print(f"  Total unique SKUs: {len(sku_counts)}")
            print(f"  Most common SKU: {sku_counts.index[0]} ({sku_counts.iloc[0]} times)")
            
            # Check for potentially invalid SKUs
            invalid_patterns = ['UNKNOWN', 'MANUAL', 'N/A', 'NULL', 'NONE', 'TBD', 'PENDING']
            found_invalid = []
            
            for pattern in invalid_patterns:
                matching_skus = [sku for sku in sku_counts.index if pattern.upper() in str(sku).upper()]
                if matching_skus:
                    found_invalid.extend(matching_skus)
            
            if found_invalid:
                print(f"\n⚠️  Potentially invalid SKUs found:")
                for sku in found_invalid:
                    print(f"  {sku}: {sku_counts[sku]} times")
            else:
                print(f"\n✅ No obviously invalid SKUs found")
            
            # Show recent entries
            print(f"\n📋 Last 10 entries:")
            recent = df.tail(10)
            for idx, row in recent.iterrows():
                sku = row.get('Confirmed_SKU', 'N/A')
                desc = row.get('Original_Description_Input', 'N/A')
                source = row.get('Source', 'N/A')
                print(f"  {sku} <- {desc[:30]}... ({source})")
                
        else:
            print(f"❌ 'Confirmed_SKU' column not found")
            
    except Exception as e:
        print(f"❌ Error reading Maestro.xlsx: {e}")

if __name__ == "__main__":
    check_current_skus()
