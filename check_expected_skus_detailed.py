import sqlite3

# Check what descriptions exist for the expected SKUs
expected_skus = {
    'DA6B51031F': 'FAROLA DERECHA',
    'DA7R510L0': 'FAROLA IZQUIERDA', 
    'DB5J500U1A': 'GUIA LATERAL IZQUIERDA PARAGOLPES DELANTERO',
    'BKB350712': 'PERSIANA',
    'DR61501T1C': 'REJILLA PARAGOLPES DELANTERO'
}

conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

print("Checking expected SKUs in database:")
print("=" * 80)

for sku, expected_desc in expected_skus.items():
    print(f"\nExpected: {expected_desc} → {sku}")
    cursor.execute("""
        SELECT vin_make, vin_year, vin_series, normalized_description, COUNT(*) as frequency
        FROM historical_parts
        WHERE sku = ?
        GROUP BY vin_make, vin_year, vin_series, normalized_description
        ORDER BY frequency DESC
    """, (sku,))
    
    results = cursor.fetchall()
    if results:
        print(f"  ✅ Found {len(results)} entries in database:")
        for make, year, series, desc, freq in results:
            print(f"    {make} {year} '{series}' - '{desc}' (freq: {freq})")
    else:
        print(f"  ❌ SKU not found in database")

print(f"\n" + "=" * 80)
print("Checking what our search terms actually find:")

# Test our current search terms vs what's in the database
search_terms = {
    'farola derecho': 'FAROLA DERECHA (after synonym expansion)',
    'farola izquierdo': 'FAROLA IZQUIERDA (after synonym expansion)',
    'guia lateral izquierdo parachoques delantero': 'GUIA LATERAL IZQUIERDA PARAGOLPES DELANTERO (after synonym expansion)',
    'deflector aire': 'PERSIANA (after synonym expansion)',
    'parrilla frontal parachoques delantero': 'REJILLA PARAGOLPES DELANTERO (after synonym expansion)'
}

for search_term, original in search_terms.items():
    print(f"\nSearching for: '{search_term}' (from {original})")
    cursor.execute("""
        SELECT DISTINCT sku, normalized_description, COUNT(*) as frequency
        FROM historical_parts
        WHERE vin_make = 'MAZDA' AND vin_year = 2015 AND vin_series LIKE '%2 [1]%' 
        AND normalized_description = ?
        GROUP BY sku, normalized_description
        ORDER BY frequency DESC
        LIMIT 5
    """, (search_term,))
    
    exact_results = cursor.fetchall()
    if exact_results:
        print(f"  ✅ Exact matches found:")
        for sku, desc, freq in exact_results:
            print(f"    {sku}: '{desc}' (freq: {freq})")
    else:
        print(f"  ❌ No exact matches")
        
        # Try fuzzy search with LIKE
        print(f"  🔍 Trying LIKE search...")
        words = search_term.split()
        like_conditions = []
        params = []
        for word in words:
            like_conditions.append("normalized_description LIKE ?")
            params.append(f'%{word}%')
        
        like_query = f"""
            SELECT DISTINCT sku, normalized_description, COUNT(*) as frequency
            FROM historical_parts
            WHERE vin_make = 'MAZDA' AND vin_year = 2015 AND vin_series LIKE '%2 [1]%' 
            AND ({' AND '.join(like_conditions)})
            GROUP BY sku, normalized_description
            ORDER BY frequency DESC
            LIMIT 5
        """
        
        cursor.execute(like_query, params)
        like_results = cursor.fetchall()
        
        if like_results:
            print(f"    Found {len(like_results)} LIKE matches:")
            for sku, desc, freq in like_results:
                print(f"      {sku}: '{desc}' (freq: {freq})")
        else:
            print(f"    No LIKE matches either")

conn.close()
print(f"\n" + "=" * 80)
print("Analysis complete!")
