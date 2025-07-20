import sqlite3

# Check what Renault data exists in the database
conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

print("=== DIAGNOSING RENAULT DATA IN DATABASE ===")
print()

# Check all makes in database
print("1. All makes in database:")
cursor.execute("SELECT DISTINCT vin_make, COUNT(*) as count FROM historical_parts GROUP BY vin_make ORDER BY count DESC")
makes = cursor.fetchall()
for make, count in makes:
    print(f"   {make}: {count} records")

print()

# Check specifically for Renault variations
print("2. Renault variations:")
cursor.execute("SELECT DISTINCT vin_make FROM historical_parts WHERE vin_make LIKE '%RENAULT%' OR vin_make LIKE '%Renault%'")
renault_makes = cursor.fetchall()
if renault_makes:
    for make in renault_makes:
        print(f"   Found: {make[0]}")
else:
    print("   No Renault entries found!")

print()

# Check years available
print("3. Years available in database:")
cursor.execute("SELECT DISTINCT vin_year, COUNT(*) as count FROM historical_parts GROUP BY vin_year ORDER BY vin_year DESC")
years = cursor.fetchall()
for year, count in years[:10]:  # Show top 10 years
    print(f"   {year}: {count} records")

print()

# Check if there's any data for 2013
print("4. Data for year 2013:")
cursor.execute("SELECT DISTINCT vin_make, COUNT(*) as count FROM historical_parts WHERE vin_year = 2013 GROUP BY vin_make ORDER BY count DESC")
data_2013 = cursor.fetchall()
if data_2013:
    for make, count in data_2013:
        print(f"   {make}: {count} records")
else:
    print("   No data for 2013!")

print()

# Check series formats
print("5. Sample series formats:")
cursor.execute("SELECT DISTINCT vin_series FROM historical_parts LIMIT 20")
series = cursor.fetchall()
for s in series:
    print(f"   '{s[0]}'")

print()

# Check if any series contains DUSTER
print("6. Series containing 'DUSTER':")
cursor.execute("SELECT DISTINCT vin_make, vin_year, vin_series, COUNT(*) as count FROM historical_parts WHERE vin_series LIKE '%DUSTER%' GROUP BY vin_make, vin_year, vin_series")
duster_data = cursor.fetchall()
if duster_data:
    for make, year, series, count in duster_data:
        print(f"   {make} {year} '{series}': {count} records")
else:
    print("   No DUSTER series found!")

print()

# Check some sample descriptions to see what parts are available
print("7. Sample part descriptions (first 10):")
cursor.execute("SELECT DISTINCT normalized_description FROM historical_parts LIMIT 10")
descriptions = cursor.fetchall()
for desc in descriptions:
    print(f"   '{desc[0]}'")

print()

# Check if any descriptions match our search terms
search_terms = [
    'absorbedor de impactos paragolpes delantero',
    'electroventilador radiador',
    'guardafango izquierdo',
    'luz antiniebla delantera derecha',
    'rejilla frontal'
]

print("8. Checking for our search terms:")
for term in search_terms:
    cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE normalized_description = ?", (term,))
    count = cursor.fetchone()[0]
    print(f"   '{term}': {count} exact matches")
    
    # Try partial matches
    cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE normalized_description LIKE ?", (f'%{term.split()[0]}%',))
    partial_count = cursor.fetchone()[0]
    print(f"   '{term.split()[0]}' (partial): {partial_count} matches")

conn.close()
print("\n=== DIAGNOSIS COMPLETE ===")
