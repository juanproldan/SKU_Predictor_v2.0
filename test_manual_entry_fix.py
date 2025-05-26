#!/usr/bin/env python3
"""
Test script to verify the manual entry fix is working correctly
"""

import pandas as pd
import os

def test_manual_entry_fix():
    """Test that manual entries are saved with actual SKUs, not 'MANUAL'"""
    
    print("🧪 Testing Manual Entry Fix")
    print("=" * 50)
    
    maestro_path = os.path.join("data", "Maestro.xlsx")
    
    if not os.path.exists(maestro_path):
        print(f"❌ Maestro.xlsx not found at {maestro_path}")
        return
    
    try:
        # Load current Maestro data
        df = pd.read_excel(maestro_path)
        
        print(f"📊 Current Maestro records: {len(df)}")
        
        # Check for any MANUAL entries (should be none after fix)
        manual_entries = df[df['Confirmed_SKU'].str.upper() == 'MANUAL']
        
        if len(manual_entries) > 0:
            print(f"🚨 STILL BROKEN: Found {len(manual_entries)} MANUAL entries!")
            print("These entries should have actual SKUs:")
            for idx, row in manual_entries.iterrows():
                print(f"  Row {idx + 2}: {row.get('Original_Description_Input', 'N/A')} -> {row.get('Confirmed_SKU', 'N/A')}")
            return False
        else:
            print(f"✅ FIXED: No MANUAL entries found!")
        
        # Check recent entries (likely the ones just added)
        recent_entries = df.tail(10)  # Last 10 entries
        
        print(f"\n📋 Recent entries (last 10):")
        print("-" * 60)
        
        for idx, row in recent_entries.iterrows():
            sku = row.get('Confirmed_SKU', 'N/A')
            desc = row.get('Original_Description_Input', 'N/A')
            source = row.get('Source', 'N/A')
            date = row.get('Date_Added', 'N/A')
            
            # Check if this looks like a manual entry
            if source == 'UserManualEntry':
                if sku == 'MANUAL':
                    print(f"  ❌ BROKEN: {desc[:40]}... -> {sku} ({source})")
                else:
                    print(f"  ✅ FIXED:  {desc[:40]}... -> {sku} ({source})")
            else:
                print(f"  ℹ️  AUTO:   {desc[:40]}... -> {sku} ({source})")
        
        # Summary
        manual_source_entries = df[df['Source'] == 'UserManualEntry']
        valid_manual_entries = manual_source_entries[manual_source_entries['Confirmed_SKU'] != 'MANUAL']
        
        print(f"\n📈 Summary:")
        print(f"  Total UserManualEntry records: {len(manual_source_entries)}")
        print(f"  Valid manual entries (not 'MANUAL'): {len(valid_manual_entries)}")
        print(f"  Broken manual entries ('MANUAL'): {len(manual_source_entries) - len(valid_manual_entries)}")
        
        if len(manual_source_entries) > 0 and len(valid_manual_entries) == len(manual_source_entries):
            print(f"  🎉 ALL MANUAL ENTRIES ARE WORKING CORRECTLY!")
            return True
        elif len(manual_source_entries) == 0:
            print(f"  ℹ️  No manual entries to test yet")
            return True
        else:
            print(f"  🚨 SOME MANUAL ENTRIES ARE STILL BROKEN!")
            return False
            
    except Exception as e:
        print(f"❌ Error reading Maestro.xlsx: {e}")
        return False

if __name__ == "__main__":
    test_manual_entry_fix()
