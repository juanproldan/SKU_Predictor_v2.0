#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from unified_consolidado_processor import get_base_path

def check_hyundai_data():
    base_path = get_base_path()
    db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🔍 Checking Hyundai data in database...")
    
    # Check Hyundai series
    print("\n📊 Hyundai series in database:")
    cursor.execute("""
        SELECT DISTINCT series, COUNT(*) as count
        FROM processed_consolidado 
        WHERE LOWER(maker) LIKE '%hyundai%' 
        GROUP BY series
        ORDER BY count DESC
        LIMIT 15
    """)
    
    for series, count in cursor.fetchall():
        print(f"  - '{series}': {count:,} records")
    
    # Check specific VIN
    print(f"\n🔍 Checking VIN: MALAT41CAJM280395")
    cursor.execute("""
        SELECT maker, model, series, COUNT(*) as count
        FROM processed_consolidado 
        WHERE vin_number = 'MALAT41CAJM280395'
        GROUP BY maker, model, series
    """)
    
    results = cursor.fetchall()
    if results:
        for maker, model, series, count in results:
            print(f"  Found: {maker} {model} {series} ({count} records)")
    else:
        print("  ❌ VIN not found in database")
    
    # Check similar VINs
    print(f"\n🔍 Checking similar VINs (MALAT41CAJ...):")
    cursor.execute("""
        SELECT vin_number, maker, model, series, COUNT(*) as count
        FROM processed_consolidado 
        WHERE vin_number LIKE 'MALAT41CAJ%'
        GROUP BY vin_number, maker, model, series
        ORDER BY count DESC
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    if results:
        for vin, maker, model, series, count in results:
            print(f"  {vin}: {maker} {model} {series} ({count} records)")
    else:
        print("  ❌ No similar VINs found")
    
    # Check Hyundai 2018 data
    print(f"\n🔍 Checking Hyundai 2018 data:")
    cursor.execute("""
        SELECT DISTINCT series, COUNT(*) as count
        FROM processed_consolidado 
        WHERE LOWER(maker) LIKE '%hyundai%' 
        AND model = '2018'
        GROUP BY series
        ORDER BY count DESC
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    if results:
        for series, count in results:
            print(f"  - '{series}': {count:,} records")
    else:
        print("  ❌ No Hyundai 2018 data found")
    
    # Check part descriptions for Hyundai
    print(f"\n🔍 Checking Hyundai part descriptions containing 'COSTADO' or 'PERSIANA':")
    cursor.execute("""
        SELECT DISTINCT descripcion, COUNT(*) as count
        FROM processed_consolidado 
        WHERE LOWER(maker) LIKE '%hyundai%' 
        AND (LOWER(descripcion) LIKE '%costado%' OR LOWER(descripcion) LIKE '%persiana%')
        GROUP BY descripcion
        ORDER BY count DESC
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    if results:
        for desc, count in results:
            print(f"  - '{desc}': {count:,} records")
    else:
        print("  ❌ No matching descriptions found")
    
    conn.close()

if __name__ == "__main__":
    check_hyundai_data()
