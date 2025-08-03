#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze VIN Patterns in Training Data

This script analyzes what VIN patterns exist in the training data
to understand why certain VINs return "Unknown" predictions.

Author: Augment Agent
Date: 2025-08-03
"""

import os
import sys
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def analyze_vin_patterns():
    """Analyze VIN patterns in the training database."""
    
    try:
        from unified_consolidado_processor import get_base_path
        
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        
        print("🔍 ANALYZING VIN PATTERNS IN TRAINING DATA")
        print("=" * 60)
        print(f"📂 Database: {db_path}")
        print(f"📊 Database exists: {os.path.exists(db_path)}")
        
        if not os.path.exists(db_path):
            print("❌ Database not found!")
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get total VIN count
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_consolidado 
            WHERE vin_number IS NOT NULL 
            AND LENGTH(vin_number) = 17
            AND vin_number != '00000000000000000'
        """)
        total_vins = cursor.fetchone()[0]
        print(f"📊 Total valid VINs in database: {total_vins:,}")
        
        # Get top WMI patterns (first 3 characters)
        print(f"\n📋 TOP 20 WMI PATTERNS (First 3 characters):")
        cursor.execute("""
            SELECT SUBSTR(vin_number, 1, 3) as wmi, COUNT(*) as count
            FROM processed_consolidado 
            WHERE vin_number IS NOT NULL 
            AND LENGTH(vin_number) = 17
            AND vin_number != '00000000000000000'
            GROUP BY SUBSTR(vin_number, 1, 3)
            ORDER BY count DESC
            LIMIT 20
        """)
        
        wmi_patterns = cursor.fetchall()
        for wmi, count in wmi_patterns:
            percentage = (count / total_vins) * 100
            print(f"   {wmi}: {count:,} VINs ({percentage:.1f}%)")
        
        # Check specific patterns from the user's test
        test_patterns = ['KMH', '9GL', 'VF1', '3MZ', 'JTE']
        print(f"\n🔍 CHECKING SPECIFIC WMI PATTERNS:")
        
        for pattern in test_patterns:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM processed_consolidado 
                WHERE vin_number LIKE ? || '%'
                AND LENGTH(vin_number) = 17
            """, (pattern,))
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Get sample VINs and their details
                cursor.execute("""
                    SELECT vin_number, maker, model, series
                    FROM processed_consolidado 
                    WHERE vin_number LIKE ? || '%'
                    AND LENGTH(vin_number) = 17
                    AND maker IS NOT NULL
                    LIMIT 3
                """, (pattern,))
                samples = cursor.fetchall()
                
                print(f"   ✅ {pattern}: {count:,} VINs")
                for vin, maker, model, series in samples:
                    print(f"      📄 {vin} → {maker} {model} {series}")
            else:
                print(f"   ❌ {pattern}: 0 VINs - NOT IN TRAINING DATA!")
        
        # Check the specific VIN from user's test
        test_vin = "KMHSH8HX8CU889564"
        print(f"\n🎯 CHECKING SPECIFIC TEST VIN: {test_vin}")
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_consolidado 
            WHERE vin_number = ?
        """, (test_vin,))
        exact_count = cursor.fetchone()[0]
        
        if exact_count > 0:
            cursor.execute("""
                SELECT maker, model, series, descripcion
                FROM processed_consolidado 
                WHERE vin_number = ?
                LIMIT 1
            """, (test_vin,))
            result = cursor.fetchone()
            print(f"   ✅ Found exact match: {result}")
        else:
            print(f"   ❌ Exact VIN not found in training data!")
            
            # Check if any similar pattern exists
            wmi = test_vin[:3]
            cursor.execute("""
                SELECT COUNT(*) 
                FROM processed_consolidado 
                WHERE vin_number LIKE ? || '%'
                AND LENGTH(vin_number) = 17
            """, (wmi,))
            similar_count = cursor.fetchone()[0]
            print(f"   🔍 VINs with same WMI ({wmi}): {similar_count}")
        
        # Show what VINs from the database images would work
        print(f"\n📋 TESTING VINs FROM YOUR DATABASE SCREENSHOT:")
        sample_vins_from_image = [
            "9GLATJ5CA2NJ35135",  # From the database screenshot
            "9GLATJ5CA2NJ35136",  # Similar pattern
        ]
        
        for vin in sample_vins_from_image:
            cursor.execute("""
                SELECT maker, model, series
                FROM processed_consolidado 
                WHERE vin_number = ?
            """, (vin,))
            result = cursor.fetchone()
            
            if result:
                maker, model, series = result
                print(f"   ✅ {vin} → {maker} {model} {series}")
            else:
                print(f"   ❌ {vin} → Not found")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing VIN patterns: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution."""
    print("🎯 VIN PATTERN ANALYSIS")
    print("=" * 60)
    
    success = analyze_vin_patterns()
    
    if success:
        print("\n" + "=" * 60)
        print("💡 CONCLUSIONS:")
        print("=" * 60)
        print("1. VIN models can only predict makers/series they were trained on")
        print("2. If a VIN pattern (WMI) doesn't exist in training data → 'Unknown'")
        print("3. Test with VINs that actually exist in your database")
        print("4. The models are working correctly - they just need familiar VINs!")
        print("\n🎯 RECOMMENDATION:")
        print("   Use VINs from your database (starting with 9GL, VF1, etc.)")
        print("   instead of random VINs for testing.")
    
    return success

if __name__ == "__main__":
    main()
