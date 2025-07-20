import sqlite3

conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

# Check MAZDA records
cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE vin_make = 'MAZDA'")
mazda_count = cursor.fetchone()[0]
print(f"MAZDA records: {mazda_count}")

# Check what makes exist
cursor.execute("SELECT DISTINCT vin_make FROM historical_parts WHERE vin_make LIKE '%MAZDA%'")
mazda_variants = cursor.fetchall()
print(f"MAZDA variants: {mazda_variants}")

# Check 2015 records
cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE vin_year = 2015")
year_2015_count = cursor.fetchone()[0]
print(f"2015 records: {year_2015_count}")

# Check series patterns
cursor.execute("SELECT DISTINCT vin_series FROM historical_parts WHERE vin_series LIKE '%2%' LIMIT 10")
series_with_2 = cursor.fetchall()
print(f"Series with '2': {series_with_2}")

# Check exact match for the predicted values
cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE vin_make = 'MAZDA' AND vin_year = 2015")
mazda_2015_count = cursor.fetchone()[0]
print(f"MAZDA 2015 records: {mazda_2015_count}")

# Check what series exist for MAZDA 2015
cursor.execute("SELECT DISTINCT vin_series FROM historical_parts WHERE vin_make = 'MAZDA' AND vin_year = 2015")
mazda_2015_series = cursor.fetchall()
print(f"MAZDA 2015 series: {mazda_2015_series}")

conn.close()
