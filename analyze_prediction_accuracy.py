#!/usr/bin/env python3
"""
Analyze prediction accuracy and identify improvement opportunities.
"""

import sys
import os
import sqlite3
import pandas as pd
sys.path.append('src')

from utils.text_utils import normalize_text

def analyze_expected_results():
    """Analyze why expected results aren't being found."""
    
    expected_results = [
        ("DA7R510L0", "FAROLA IZQUIERDA"),
        ("DB1T56141A", "GUARDAPOLVO PLASTICO DELANTERO IZQUIERDO"),
        ("DB5J507K0", "BOCEL IZQUIERDO PERSIANA - CROMADO"),
        ("DB9L61439", "CALCOMANIA CAPO A/A"),
        ("PE0115031", "CALCOMANIA CAPO PRECAUCION"),
        ("DB5J500U1B", "GUIA LATERAL IZQUIERDA PARAGOLPES DELANTERO"),
        ("TK2150355", "REMACHE - Remache negro de persiana")
    ]
    
    print("=== ANALYZING EXPECTED RESULTS ===\n")
    
    # Check database
    db_path = 'data/consolidado.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        
        for sku, description in expected_results:
            print(f"🔍 Analyzing: {sku} - {description}")
            
            # Normalize the description
            normalized_desc = normalize_text(description, expand_linguistic_variations=True)
            print(f"   Normalized: {normalized_desc}")
            
            # Check if SKU exists in database
            sku_query = "SELECT * FROM filtered_bids WHERE item_sku = ?"
            sku_results = pd.read_sql_query(sku_query, conn, params=[sku])

            if len(sku_results) > 0:
                print(f"   ✅ SKU found in database:")
                for _, row in sku_results.iterrows():
                    print(f"      Description: {row.get('item_original_description', 'N/A')}")
                    print(f"      Make: {row.get('vin_make', 'N/A')}")
                    print(f"      Year: {row.get('vin_year', 'N/A')}")
                    print(f"      Series: {row.get('vin_series', 'N/A')}")
            else:
                print(f"   ❌ SKU NOT found in database")

            # Check if description exists (fuzzy search)
            desc_query = """
                SELECT item_sku, item_original_description, vin_make, vin_year, vin_series
                FROM filtered_bids
                WHERE item_original_description LIKE ?
                LIMIT 5
            """
            desc_results = pd.read_sql_query(desc_query, conn, params=[f"%{normalized_desc}%"])
            
            if len(desc_results) > 0:
                print(f"   📋 Similar descriptions found:")
                for _, row in desc_results.iterrows():
                    print(f"      {row['item_sku']}: {row['item_original_description']}")
            else:
                print(f"   ❌ No similar descriptions found")
            
            print("-" * 60)
        
        conn.close()
    else:
        print("❌ Database not found")
    
    # Check Maestro data
    maestro_path = 'data/Maestro.xlsx'
    if os.path.exists(maestro_path):
        print("\n=== CHECKING MAESTRO DATA ===")
        df = pd.read_excel(maestro_path)
        
        for sku, description in expected_results:
            normalized_desc = normalize_text(description, expand_linguistic_variations=True)
            
            # Check if this description is in Maestro
            maestro_matches = df[
                df['Normalized_Description_Input'].str.lower().str.contains(
                    normalized_desc.lower(), na=False
                )
            ]
            
            if len(maestro_matches) > 0:
                print(f"✅ Found in Maestro: {description}")
                for _, row in maestro_matches.iterrows():
                    print(f"   SKU: {row.get('Confirmed_SKU', 'N/A')}")
            else:
                print(f"❌ Not in Maestro: {description}")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Check if expected SKUs exist in the database")
    print("2. Verify text normalization is working correctly")
    print("3. Check if VIN prediction is accurate for vehicle details")
    print("4. Consider expanding training data")
    print("5. Review confidence scoring algorithm")

if __name__ == "__main__":
    analyze_expected_results()
