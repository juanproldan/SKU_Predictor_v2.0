import sqlite3

# Check the exact VIN prediction vs database
conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

# From terminal output - what the VIN predictor gives us
predicted_make = "RENAULT"  # From search logs
predicted_year = 2013
predicted_series = "RENAULT/DUSTER (HS)/BASICO"

print("=== VIN PREDICTION vs DATABASE MISMATCH ===")
print(f"VIN Predictor gives: Make='{predicted_make}', Year={predicted_year}, Series='{predicted_series}'")
print()

# Check exact match with predicted values
cursor.execute("""
    SELECT COUNT(*) 
    FROM historical_parts 
    WHERE vin_make = ? AND vin_year = ? AND vin_series = ?
""", (predicted_make, predicted_year, predicted_series))
exact_count = cursor.fetchone()[0]
print(f"1. Exact match with predicted values: {exact_count} records")

# Check with correct case (from our previous analysis)
correct_make = "Renault"  # From database analysis
cursor.execute("""
    SELECT COUNT(*) 
    FROM historical_parts 
    WHERE vin_make = ? AND vin_year = ? AND vin_series = ?
""", (correct_make, predicted_year, predicted_series))
correct_count = cursor.fetchone()[0]
print(f"2. Match with correct case 'Renault': {correct_count} records")

if correct_count > 0:
    print(f"\n✅ FOUND THE ISSUE! Database uses '{correct_make}' not '{predicted_make}'")
    
    # Check what descriptions are available with correct make
    print(f"\n3. Available descriptions with correct make '{correct_make}':")
    cursor.execute("""
        SELECT DISTINCT normalized_description, COUNT(*) as count
        FROM historical_parts 
        WHERE vin_make = ? AND vin_year = ? AND vin_series = ?
        GROUP BY normalized_description
        ORDER BY count DESC
        LIMIT 20
    """, (correct_make, predicted_year, predicted_series))
    descriptions = cursor.fetchall()
    
    for desc, count in descriptions:
        print(f"     '{desc}': {count} records")
    
    # Test our abbreviated search terms
    print(f"\n4. Testing our abbreviated terms with correct make:")
    test_terms = [
        'absorbimpacto paragolpes del',
        'electrovent radiador',
        'guardafango i',
        'luz antiniebla del d',
        'luz antiniebla del i'
    ]
    
    for term in test_terms:
        cursor.execute("""
            SELECT COUNT(*), GROUP_CONCAT(DISTINCT sku) as skus
            FROM historical_parts 
            WHERE vin_make = ? AND vin_year = ? AND vin_series = ? AND normalized_description = ?
        """, (correct_make, predicted_year, predicted_series, term))
        result = cursor.fetchone()
        count, skus = result[0], result[1]
        print(f"     '{term}': {count} matches")
        if skus:
            print(f"       SKUs: {skus}")

else:
    print(f"\n❌ Still no match with correct case. Let's check all make variations:")
    cursor.execute("""
        SELECT DISTINCT vin_make, COUNT(*) as count
        FROM historical_parts 
        WHERE vin_year = ? AND vin_series = ?
        GROUP BY vin_make
        ORDER BY count DESC
    """, (predicted_year, predicted_series))
    make_variations = cursor.fetchall()
    
    for make, count in make_variations:
        print(f"     '{make}': {count} records")

conn.close()
print("\n=== DIAGNOSIS COMPLETE ===")
