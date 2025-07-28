import sqlite3

# Check database content
conn = sqlite3.connect('Source_Files/processed_consolidado.db')
cursor = conn.cursor()

# Check for our specific test parts
test_parts = ['pegante vidrio 2', 'panoramico delantero', 'empaque vidrio panoramico']

for part in test_parts:
    print(f'\n--- Searching for: {part} ---')

    # Exact match
    cursor.execute('SELECT referencia, original_descripcion, normalized_descripcion FROM processed_consolidado WHERE normalized_descripcion = ? LIMIT 5', (part,))
    exact_matches = cursor.fetchall()
    print(f'Exact matches: {len(exact_matches)}')
    for match in exact_matches:
        print(f'  referencia: {match[0]}, Original: {match[1]}, Normalized: {match[2]}')

    # Partial match
    cursor.execute('SELECT referencia, original_descripcion, normalized_descripcion FROM processed_consolidado WHERE normalized_descripcion LIKE ? LIMIT 5', (f'%{part}%',))
    partial_matches = cursor.fetchall()
    print(f'Partial matches: {len(partial_matches)}')
    for match in partial_matches[:3]:  # Show first 3
        print(f'  referencia: {match[0]}, Original: {match[1]}, Normalized: {match[2]}')

# Check for vidrio parts in general
print(f'\n--- General vidrio parts ---')
cursor.execute('SELECT referencia, original_descripcion, normalized_descripcion FROM processed_consolidado WHERE normalized_descripcion LIKE "%vidrio%" LIMIT 10')
vidrio_parts = cursor.fetchall()
print(f'Vidrio parts found: {len(vidrio_parts)}')
for part in vidrio_parts[:5]:
    print(f'  referencia: {part[0]}, Original: {part[1]}, Normalized: {part[2]}')

conn.close()
