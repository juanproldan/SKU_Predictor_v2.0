import sqlite3

# Check what descriptions exist for the expected SKUs
expected_skus = ['DA3651031F', 'DA7R51010', 'DB5J500U1A', 'BKB350712', 'DR61501T1C']

conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

print("Checking expected SKUs in database:")
print("=" * 50)

for sku in expected_skus:
    print(f"\nSKU: {sku}")
    cursor.execute("""
        SELECT vin_make, vin_year, vin_series, normalized_description, COUNT(*) as frequency
        FROM historical_parts
        WHERE sku = ?
        GROUP BY vin_make, vin_year, vin_series, normalized_description
        ORDER BY frequency DESC
    """, (sku,))
    
    results = cursor.fetchall()
    if results:
        print(f"  Found {len(results)} entries:")
        for make, year, series, desc, freq in results:
            print(f"    {make} {year} '{series}' - '{desc}' (freq: {freq})")
    else:
        print(f"  ❌ SKU not found in database")

# Check what MAZDA 2015 descriptions exist that are similar to our search terms
print(f"\n" + "=" * 50)
print("MAZDA 2015 descriptions containing key terms:")

search_terms = ['farola', 'persiana', 'rejilla', 'paragolpes']
for term in search_terms:
    print(f"\nTerm: '{term}'")
    cursor.execute("""
        SELECT DISTINCT normalized_description, COUNT(*) as frequency
        FROM historical_parts
        WHERE vin_make = 'MAZDA' AND vin_year = 2015 AND normalized_description LIKE ?
        GROUP BY normalized_description
        ORDER BY frequency DESC
        LIMIT 5
    """, (f'%{term}%',))
    
    results = cursor.fetchall()
    for desc, freq in results:
        print(f"  '{desc}' (freq: {freq})")

conn.close()
