import sqlite3

# Test the fuzzy series matching fix
conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

# Test parameters from the VIN prediction
predicted_make = 'MAZDA'
predicted_year = 2015
predicted_series = '2 [1]'
test_description = 'paragolpes delantero'

print(f"Testing fuzzy series matching fix:")
print(f"Predicted: Make='{predicted_make}', Year={predicted_year}, Series='{predicted_series}'")
print(f"Test description: '{test_description}'")
print()

# Test 1: Exact series match (should fail)
print("1. Exact series match:")
cursor.execute("""
    SELECT sku, COUNT(*) as frequency
    FROM historical_parts
    WHERE vin_make = ? AND vin_year = ? AND vin_series = ? AND normalized_description = ?
    GROUP BY sku
""", (predicted_make, predicted_year, predicted_series, test_description))
exact_results = cursor.fetchall()
print(f"   Results: {len(exact_results)} SKUs found")
for sku, freq in exact_results:
    print(f"   - {sku} (freq: {freq})")

# Test 2: Fuzzy series match (should work)
print("\n2. Fuzzy series match (LIKE):")
cursor.execute("""
    SELECT sku, COUNT(*) as frequency
    FROM historical_parts
    WHERE vin_make = ? AND vin_year = ? AND vin_series LIKE ? AND normalized_description = ?
    GROUP BY sku
""", (predicted_make, predicted_year, f'%{predicted_series}%', test_description))
fuzzy_results = cursor.fetchall()
print(f"   Results: {len(fuzzy_results)} SKUs found")
for sku, freq in fuzzy_results:
    print(f"   - {sku} (freq: {freq})")

# Test 3: Show what series actually exist for MAZDA 2015
print(f"\n3. Available series for {predicted_make} {predicted_year}:")
cursor.execute("""
    SELECT DISTINCT vin_series, COUNT(*) as count
    FROM historical_parts
    WHERE vin_make = ? AND vin_year = ?
    GROUP BY vin_series
    ORDER BY count DESC
""", (predicted_make, predicted_year))
series_results = cursor.fetchall()
for series, count in series_results:
    print(f"   - '{series}' ({count} records)")

# Test 4: Check if the description exists for MAZDA 2015
print(f"\n4. Available descriptions for {predicted_make} {predicted_year} containing '{test_description}':")
cursor.execute("""
    SELECT DISTINCT normalized_description, COUNT(*) as count
    FROM historical_parts
    WHERE vin_make = ? AND vin_year = ? AND normalized_description LIKE ?
    GROUP BY normalized_description
    ORDER BY count DESC
    LIMIT 5
""", (predicted_make, predicted_year, f'%{test_description}%'))
desc_results = cursor.fetchall()
for desc, count in desc_results:
    print(f"   - '{desc}' ({count} records)")

conn.close()
print("\n✅ Fuzzy series matching test complete!")
