import sqlite3

# Check database tables and schema
conn = sqlite3.connect('Fixacar_SKU_Predictor_CLIENT/Source_Files/processed_consolidado.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Tables:", tables)

# Check processed_consolidado table schema if it exists
if 'processed_consolidado' in tables:
    cursor.execute("PRAGMA table_info(processed_consolidado)")
    columns = [row[1] for row in cursor.fetchall()]
    print("processed_consolidado columns:", columns)

# Check year range tables if they exist
if 'sku_year_ranges' in tables:
    cursor.execute("PRAGMA table_info(sku_year_ranges)")
    columns = [row[1] for row in cursor.fetchall()]
    print("sku_year_ranges columns:", columns)

conn.close()
