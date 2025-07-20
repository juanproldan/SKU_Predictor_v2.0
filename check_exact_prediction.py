import sqlite3

# Check exact match for our predicted vehicle
conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

# Our VIN prediction results
predicted_make = "Renault"  # From terminal output
predicted_year = 2013       # From terminal output  
predicted_series = "RENAULT/DUSTER (HS)/BASICO"  # From terminal output

print("=== CHECKING EXACT PREDICTION MATCH ===")
print(f"Predicted: Make='{predicted_make}', Year={predicted_year}, Series='{predicted_series}'")
print()

# Check exact match
print("1. Exact match check:")
cursor.execute("""
    SELECT COUNT(*) 
    FROM historical_parts 
    WHERE vin_make = ? AND vin_year = ? AND vin_series = ?
""", (predicted_make, predicted_year, predicted_series))
exact_count = cursor.fetchone()[0]
print(f"   Exact match: {exact_count} records")

if exact_count == 0:
    print("   ❌ No exact match found!")
    
    # Check each component separately
    print("\n2. Component-wise check:")
    
    # Check make variations
    cursor.execute("SELECT DISTINCT vin_make FROM historical_parts WHERE vin_make LIKE ?", (f'%{predicted_make}%',))
    make_variations = cursor.fetchall()
    print(f"   Make variations for '{predicted_make}':")
    for make in make_variations:
        cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE vin_make = ?", (make[0],))
        count = cursor.fetchone()[0]
        print(f"     '{make[0]}': {count} records")
    
    # Check year
    cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE vin_year = ?", (predicted_year,))
    year_count = cursor.fetchone()[0]
    print(f"   Year {predicted_year}: {year_count} records")
    
    # Check series variations
    cursor.execute("SELECT DISTINCT vin_series FROM historical_parts WHERE vin_series LIKE ?", (f'%DUSTER%BASICO%',))
    series_variations = cursor.fetchall()
    print(f"   Series variations containing 'DUSTER' and 'BASICO':")
    for series in series_variations[:10]:  # Show first 10
        cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE vin_series = ?", (series[0],))
        count = cursor.fetchone()[0]
        print(f"     '{series[0]}': {count} records")

else:
    print("   ✅ Exact match found!")
    
    # Check what descriptions are available for this exact match
    print("\n3. Available descriptions for exact match:")
    cursor.execute("""
        SELECT DISTINCT normalized_description, COUNT(*) as count
        FROM historical_parts 
        WHERE vin_make = ? AND vin_year = ? AND vin_series = ?
        GROUP BY normalized_description
        ORDER BY count DESC
        LIMIT 20
    """, (predicted_make, predicted_year, predicted_series))
    descriptions = cursor.fetchall()
    
    for desc, count in descriptions:
        print(f"     '{desc}': {count} records")

# Check our specific search terms with the predicted vehicle
print(f"\n4. Checking our search terms with predicted vehicle:")
search_terms = [
    'absorbedor de impactos paragolpes delantero',
    'electroventilador radiador', 
    'guardafango izquierdo',
    'luz antiniebla delantera derecha',
    'luz antiniebla delantera izquierda',
    'rejilla frontal'
]

for term in search_terms:
    cursor.execute("""
        SELECT COUNT(*), GROUP_CONCAT(DISTINCT sku) as skus
        FROM historical_parts 
        WHERE vin_make = ? AND vin_year = ? AND vin_series = ? AND normalized_description = ?
    """, (predicted_make, predicted_year, predicted_series, term))
    result = cursor.fetchone()
    count, skus = result[0], result[1]
    print(f"   '{term}': {count} matches")
    if skus:
        print(f"     SKUs: {skus}")

conn.close()
print("\n=== CHECK COMPLETE ===")
