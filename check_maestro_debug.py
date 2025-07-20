#!/usr/bin/env python3
"""
Debug script to check Maestro data and search logic
"""

import pandas as pd
import os

def check_maestro_data():
    maestro_path = 'data/Maestro.xlsx'
    if not os.path.exists(maestro_path):
        print('❌ Maestro.xlsx not found')
        return
    
    df = pd.read_excel(maestro_path)
    print('=== MAESTRO DATA ANALYSIS ===')
    print(f'Total records: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    print()
    
    # Show last 5 records (most recent saves)
    print('=== LAST 5 RECORDS (Most Recent) ===')
    for idx, row in df.tail(5).iterrows():
        print(f'Row {idx + 2}:')
        print(f'  VIN_Make: {row.get("VIN_Make", "N/A")}')
        print(f'  VIN_Year_Min: {row.get("VIN_Year_Min", "N/A")} (type: {type(row.get("VIN_Year_Min", "N/A"))})')
        print(f'  VIN_Series_Trim: {row.get("VIN_Series_Trim", "N/A")}')
        print(f'  Original_Description: {row.get("Original_Description_Input", "N/A")}')
        print(f'  Normalized_Description: {row.get("Normalized_Description_Input", "N/A")}')
        print(f'  Confirmed_SKU: {row.get("Confirmed_SKU", "N/A")}')
        print()

def simulate_search():
    """Simulate the search that should find the saved data"""
    print('=== SIMULATING SEARCH LOGIC ===')
    
    # Test data from your VIN: PF8HC8AGDM015033
    test_vin_make = "RENAULT"
    test_vin_year = "2013"  # String format
    test_vin_series = "RENAULT/DUSTER (HS)/BASICO"
    test_descriptions = [
        "absorbedor de impactos paragolpes delantero",
        "electroventilador radiador", 
        "guardafango izquierdo",
        "luz antiniebla delantera derecha",
        "luz antiniebla delantera izquierda"
    ]
    
    maestro_path = 'data/Maestro.xlsx'
    if not os.path.exists(maestro_path):
        print('❌ Maestro.xlsx not found')
        return
    
    df = pd.read_excel(maestro_path)
    
    print(f'Searching for matches with:')
    print(f'  VIN_Make: {test_vin_make}')
    print(f'  VIN_Year: {test_vin_year}')
    print(f'  VIN_Series: {test_vin_series}')
    print()
    
    for desc in test_descriptions:
        print(f'Searching for description: "{desc}"')
        
        # Check exact matches
        matches = df[
            (df['VIN_Make'].astype(str).str.upper() == test_vin_make) &
            (df['VIN_Year_Min'].astype(str) == test_vin_year) &
            (df['VIN_Series_Trim'].astype(str).str.upper() == test_vin_series.upper()) &
            (df['Normalized_Description_Input'].astype(str).str.lower() == desc.lower())
        ]
        
        if len(matches) > 0:
            print(f'  ✅ Found {len(matches)} exact matches!')
            for idx, match in matches.iterrows():
                print(f'    SKU: {match.get("Confirmed_SKU", "N/A")}')
        else:
            print(f'  ❌ No exact matches found')
            
            # Check partial matches for debugging
            make_matches = df[df['VIN_Make'].astype(str).str.upper() == test_vin_make]
            print(f'    Make matches: {len(make_matches)}')
            
            year_matches = df[df['VIN_Year_Min'].astype(str) == test_vin_year]
            print(f'    Year matches: {len(year_matches)}')
            
            series_matches = df[df['VIN_Series_Trim'].astype(str).str.upper() == test_vin_series.upper()]
            print(f'    Series matches: {len(series_matches)}')
            
            desc_matches = df[df['Normalized_Description_Input'].astype(str).str.lower() == desc.lower()]
            print(f'    Description matches: {len(desc_matches)}')
        
        print()

if __name__ == "__main__":
    check_maestro_data()
    simulate_search()
