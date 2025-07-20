import sqlite3

conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

# Check exact match for predicted values
predicted_make = 'MAZDA'
predicted_year = 2015
predicted_series = '2 [1]'

print(f"Looking for: Make='{predicted_make}', Year={predicted_year}, Series='{predicted_series}'")

# Exact match
cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE vin_make = ? AND vin_year = ? AND vin_series = ?", 
               (predicted_make, predicted_year, predicted_series))
exact_count = cursor.fetchone()[0]
print(f"Exact match count: {exact_count}")

# Check similar series
cursor.execute("SELECT DISTINCT vin_series FROM historical_parts WHERE vin_make = ? AND vin_year = ? AND vin_series LIKE ?", 
               (predicted_make, predicted_year, '%2 [1]%'))
similar_series = cursor.fetchall()
print(f"Similar series: {similar_series}")

# Check if there are any records with the exact series anywhere
cursor.execute("SELECT COUNT(*) FROM historical_parts WHERE vin_series = ?", (predicted_series,))
series_anywhere = cursor.fetchone()[0]
print(f"Series '{predicted_series}' exists anywhere: {series_anywhere}")

# Check what descriptions exist for MAZDA 2015
cursor.execute("SELECT DISTINCT normalized_description FROM historical_parts WHERE vin_make = ? AND vin_year = ? LIMIT 10", 
               (predicted_make, predicted_year))
descriptions = cursor.fetchall()
print(f"Sample descriptions for MAZDA 2015: {descriptions}")

conn.close()
