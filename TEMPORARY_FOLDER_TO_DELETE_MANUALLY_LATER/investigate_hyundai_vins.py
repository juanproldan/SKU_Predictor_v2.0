#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Investigate Hyundai VINs Issue

Let's check if the Hyundai VINs from the user's screenshot actually exist
in the training data and why they might be failing.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def investigate_hyundai_vins():
    """Investigate the specific Hyundai VINs from the user's screenshot."""
    
    try:
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        
        print("🔍 INVESTIGATING HYUNDAI VINs FROM YOUR SCREENSHOT")
        print("=" * 60)
        print(f"📂 Database: {db_path}")
        
        if not os.path.exists(db_path):
            print("❌ Database not found!")
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # The VIN from your test
        test_vin = "KMHSH8HX8CU889564"
        
        print(f"🎯 CHECKING YOUR TEST VIN: {test_vin}")
        print("=" * 60)
        
        # Check if this exact VIN exists
        cursor.execute("""
            SELECT vin_number, maker, model, series, descripcion
            FROM processed_consolidado 
            WHERE vin_number = ?
        """, (test_vin,))
        
        exact_match = cursor.fetchone()
        if exact_match:
            vin, maker, model, series, desc = exact_match
            print(f"✅ EXACT VIN FOUND IN DATABASE!")
            print(f"   VIN: {vin}")
            print(f"   Maker: {maker}")
            print(f"   Model: {model}")
            print(f"   Series: {series}")
            print(f"   Description: {desc[:100]}...")
        else:
            print(f"❌ EXACT VIN NOT FOUND IN DATABASE")
        
        # Check if similar VINs exist (same WMI pattern)
        wmi = test_vin[:3]  # KMH
        print(f"\n🔍 CHECKING VINs WITH SAME WMI PATTERN: {wmi}")
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_consolidado 
            WHERE vin_number LIKE ? || '%'
            AND LENGTH(vin_number) = 17
        """, (wmi,))
        
        wmi_count = cursor.fetchone()[0]
        print(f"   VINs with WMI '{wmi}': {wmi_count:,}")
        
        if wmi_count > 0:
            # Get some examples
            cursor.execute("""
                SELECT vin_number, maker, model, series
                FROM processed_consolidado 
                WHERE vin_number LIKE ? || '%'
                AND LENGTH(vin_number) = 17
                AND maker IS NOT NULL
                LIMIT 5
            """, (wmi,))
            
            examples = cursor.fetchall()
            print(f"   📋 Sample VINs with WMI '{wmi}':")
            for vin, maker, model, series in examples:
                print(f"      {vin} → {maker} {model} {series}")
        
        # Check the specific VDS pattern from your VIN
        vds_pattern = test_vin[3:8]  # SH8HX
        print(f"\n🔍 CHECKING VDS PATTERN: {vds_pattern}")
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_consolidado 
            WHERE vin_number LIKE ? || ? || '%'
            AND LENGTH(vin_number) = 17
        """, (wmi, vds_pattern))
        
        vds_count = cursor.fetchone()[0]
        print(f"   VINs with WMI+VDS '{wmi}{vds_pattern}': {vds_count}")
        
        # Now let's check what happens during training
        print(f"\n🧪 CHECKING TRAINING DATA FILTERING")
        print("=" * 60)
        
        # Check what the training script actually uses
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_consolidado 
            WHERE vin_number IS NOT NULL 
            AND LENGTH(vin_number) = 17
            AND vin_number != '00000000000000000'
            AND maker IS NOT NULL 
            AND model IS NOT NULL 
            AND series IS NOT NULL
            AND vin_number LIKE ? || '%'
        """, (wmi,))
        
        training_count = cursor.fetchone()[0]
        print(f"   Hyundai VINs that meet training criteria: {training_count:,}")
        
        # Check if your specific VIN meets training criteria
        cursor.execute("""
            SELECT vin_number, maker, model, series
            FROM processed_consolidado 
            WHERE vin_number = ?
            AND vin_number IS NOT NULL 
            AND LENGTH(vin_number) = 17
            AND vin_number != '00000000000000000'
            AND maker IS NOT NULL 
            AND model IS NOT NULL 
            AND series IS NOT NULL
        """, (test_vin,))
        
        training_match = cursor.fetchone()
        if training_match:
            print(f"✅ YOUR VIN MEETS TRAINING CRITERIA!")
            print(f"   This VIN should have been included in training")
        else:
            print(f"❌ YOUR VIN DOES NOT MEET TRAINING CRITERIA")
            print(f"   Checking why...")
            
            # Check each criterion individually
            cursor.execute("SELECT vin_number, maker, model, series FROM processed_consolidado WHERE vin_number = ?", (test_vin,))
            raw_data = cursor.fetchone()
            if raw_data:
                vin, maker, model, series = raw_data
                print(f"   Raw data: VIN={vin}, Maker={maker}, Model={model}, Series={series}")
                print(f"   Issues:")
                if not vin or len(vin) != 17:
                    print(f"     - VIN length issue: {len(vin) if vin else 'NULL'}")
                if vin == '00000000000000000':
                    print(f"     - VIN is all zeros")
                if not maker:
                    print(f"     - Maker is NULL")
                if not model:
                    print(f"     - Model is NULL")
                if not series:
                    print(f"     - Series is NULL")
        
        # Let's also check what VINs from your screenshot actually exist
        print(f"\n🔍 CHECKING VINs FROM YOUR SCREENSHOT")
        print("=" * 60)
        
        # VINs that appear to be in your screenshot (starting with 9GLATJ5CA2NJ)
        screenshot_pattern = "9GLATJ5CA2NJ"
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_consolidado 
            WHERE vin_number LIKE ? || '%'
        """, (screenshot_pattern,))
        
        screenshot_count = cursor.fetchone()[0]
        print(f"   VINs starting with '{screenshot_pattern}': {screenshot_count}")
        
        if screenshot_count > 0:
            cursor.execute("""
                SELECT vin_number, maker, model, series
                FROM processed_consolidado 
                WHERE vin_number LIKE ? || '%'
                LIMIT 3
            """, (screenshot_pattern,))
            
            screenshot_examples = cursor.fetchall()
            print(f"   📋 Examples:")
            for vin, maker, model, series in screenshot_examples:
                print(f"      {vin} → {maker} {model} {series}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error investigating VINs: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    investigate_hyundai_vins()
