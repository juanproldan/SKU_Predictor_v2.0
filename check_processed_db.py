import sqlite3

# Connect to the database
conn = sqlite3.connect('Processed_Data.db')
cursor = conn.cursor()

# Get list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:")
for table in tables:
    print(f"  - {table[0]}")

# For each table, get the schema
for table in tables:
    table_name = table[0]
    print(f"\nSchema for table '{table_name}':")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for column in columns:
        print(f"  - {column[1]} ({column[2]})")

# Get row counts for each table
for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"\nRow count for table '{table_name}': {count}")

# Sample data from each table
for table in tables:
    table_name = table[0]
    print(f"\nSample data from table '{table_name}':")
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
    columns = [description[0] for description in cursor.description]
    print("Columns:", columns)
    
    rows = cursor.fetchall()
    for row in rows:
        print("Row:", row)

# Close the connection
conn.close()
