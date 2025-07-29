import pandas as pd

# Examine Text Processing Rules
try:
    df = pd.read_excel('Source_Files/Text_Processing_Rules.xlsx', sheet_name=None)
    print('Available sheets:')
    print(list(df.keys()))
    
    for sheet_name, sheet_df in df.items():
        print(f'\n{sheet_name} sheet:')
        print(f'Columns: {sheet_df.columns.tolist()}')
        print(f'Rows: {len(sheet_df)}')
        if len(sheet_df) > 0:
            print('Sample data:')
            print(sheet_df.head(2).to_string())
except Exception as e:
    print(f"Error reading Text_Processing_Rules.xlsx: {e}")

print("\n" + "="*50)

# Examine Consolidado.json structure
try:
    import json
    with open('Source_Files/Consolidado.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Consolidado.json structure:")
    print(f"Type: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"Total records: {len(data)}")
        print("Sample record keys:")
        print(list(data[0].keys()))
        print("\nFirst record:")
        for k, v in data[0].items():
            print(f"  {k}: {v}")
except Exception as e:
    print(f"Error reading Consolidado.json: {e}")
