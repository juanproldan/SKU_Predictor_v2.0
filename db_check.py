import sqlite3

conn = sqlite3.connect('Source_Files/processed_consolidado.db')
cursor = conn.cursor()

print("Database Schema:")
cursor.execute("PRAGMA table_info(processed_consolidado)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]} ({col[2]})")

print("\nAvailable Mazda Series:")
cursor.execute("SELECT DISTINCT maker, series FROM processed_consolidado WHERE UPPER(maker) = 'MAZDA' LIMIT 10")
results = cursor.fetchall()
for maker, series in results:
    print(f"  {maker}: {series}")

print("\nSample Descriptions:")
cursor.execute("SELECT DISTINCT descripcion FROM processed_consolidado WHERE UPPER(maker) = 'MAZDA' LIMIT 5")
descriptions = cursor.fetchall()
for desc in descriptions:
    print(f"  {desc[0]}")

print("\nSearching for 'paragolpes' descriptions:")
cursor.execute("SELECT maker, series, descripcion, referencia FROM processed_consolidado WHERE UPPER(maker) = 'MAZDA' AND UPPER(descripcion) LIKE '%PARAGOLPES%' LIMIT 5")
paragolpes_results = cursor.fetchall()
for maker, series, desc, ref in paragolpes_results:
    print(f"  {maker} {series}: {desc} -> {ref}")

print("\nSearching for 'bumper' descriptions:")
cursor.execute("SELECT maker, series, descripcion, referencia FROM processed_consolidado WHERE UPPER(maker) = 'MAZDA' AND UPPER(descripcion) LIKE '%BUMPER%' LIMIT 5")
bumper_results = cursor.fetchall()
for maker, series, desc, ref in bumper_results:
    print(f"  {maker} {series}: {desc} -> {ref}")

print("\nTesting dual matching strategy:")
# Test 1: Exact match for "paragolpes delantero"
test_descriptions = [
    "paragolpes delantero",
    "PARAGOLPES DELANTERO",
    "guia lateral derecha paragolpes delantero",
    "GUIA LATERAL DERECHA PARAGOLPES DELANTERO"
]

for desc in test_descriptions:
    print(f"\n  Testing: '{desc}'")
    # Try both original and normalized versions
    cursor.execute("SELECT maker, series, descripcion, referencia FROM processed_consolidado WHERE UPPER(maker) = 'MAZDA' AND UPPER(descripcion) = UPPER(?) LIMIT 3", (desc,))
    results = cursor.fetchall()
    if results:
        for maker, series, db_desc, ref in results:
            print(f"    MATCH: {maker} {series}: {db_desc} -> {ref}")
    else:
        print(f"    No exact matches found")

conn.close()
