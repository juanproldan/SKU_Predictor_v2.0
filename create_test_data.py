#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create realistic test data to demonstrate year range functionality
"""

import sqlite3
import os
import sys

# Add src to path
sys.path.append('src')

def get_base_path():
    """Get the base path for the application."""
    return os.path.join(os.getcwd(), "Fixacar_SKU_Predictor_CLIENT")

def create_test_data():
    """Create realistic test data for year range demonstration."""
    
    db_path = os.path.join(get_base_path(), "Source_Files", "processed_consolidado.db")
    
    print(f"Creating test data in: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("DELETE FROM processed_consolidado")
    cursor.execute("DELETE FROM sku_year_ranges")
    cursor.execute("DELETE FROM vin_year_ranges")
    
    # Create realistic automotive test data
    test_data = [
        # Toyota Corolla parts across multiple years (should create year ranges)
        ("VIN001", "Toyota", 2018, "Corolla", "filtro aire", "filtro aire", "TOY-AIR-001"),
        ("VIN002", "Toyota", 2019, "Corolla", "filtro aire", "filtro aire", "TOY-AIR-001"),
        ("VIN003", "Toyota", 2020, "Corolla", "filtro aire", "filtro aire", "TOY-AIR-001"),
        ("VIN004", "Toyota", 2021, "Corolla", "filtro aire", "filtro aire", "TOY-AIR-001"),
        ("VIN005", "Toyota", 2022, "Corolla", "filtro aire", "filtro aire", "TOY-AIR-001"),
        
        # Same part but different description variations
        ("VIN006", "Toyota", 2019, "Corolla", "filtro de aire", "filtro aire", "TOY-AIR-001"),
        ("VIN007", "Toyota", 2020, "Corolla", "filtro de aire motor", "filtro aire", "TOY-AIR-001"),
        
        # Different part for same vehicle
        ("VIN008", "Toyota", 2018, "Corolla", "pastillas freno", "pastillas freno", "TOY-BRAKE-001"),
        ("VIN009", "Toyota", 2019, "Corolla", "pastillas freno", "pastillas freno", "TOY-BRAKE-001"),
        ("VIN010", "Toyota", 2020, "Corolla", "pastillas freno", "pastillas freno", "TOY-BRAKE-001"),
        
        # Mazda CX-5 parts (different make/series)
        ("VIN011", "Mazda", 2017, "CX-5", "filtro aire", "filtro aire", "MAZ-AIR-001"),
        ("VIN012", "Mazda", 2018, "CX-5", "filtro aire", "filtro aire", "MAZ-AIR-001"),
        ("VIN013", "Mazda", 2019, "CX-5", "filtro aire", "filtro aire", "MAZ-AIR-001"),
        ("VIN014", "Mazda", 2020, "CX-5", "filtro aire", "filtro aire", "MAZ-AIR-001"),
        
        # Ford Focus parts (another make/series)
        ("VIN015", "Ford", 2016, "Focus", "filtro combustible", "filtro combustible", "FORD-FUEL-001"),
        ("VIN016", "Ford", 2017, "Focus", "filtro combustible", "filtro combustible", "FORD-FUEL-001"),
        ("VIN017", "Ford", 2018, "Focus", "filtro combustible", "filtro combustible", "FORD-FUEL-001"),
        
        # Some parts with gaps (should still create single range)
        ("VIN018", "Chevrolet", 2015, "Cruze", "amortiguador delantero", "amortiguador delantero", "CHEV-SHOCK-001"),
        ("VIN019", "Chevrolet", 2016, "Cruze", "amortiguador delantero", "amortiguador delantero", "CHEV-SHOCK-001"),
        # Gap in 2017
        ("VIN020", "Chevrolet", 2018, "Cruze", "amortiguador delantero", "amortiguador delantero", "CHEV-SHOCK-001"),
        ("VIN021", "Chevrolet", 2019, "Cruze", "amortiguador delantero", "amortiguador delantero", "CHEV-SHOCK-001"),
        
        # High frequency part (should get high confidence)
        ("VIN022", "Toyota", 2020, "Corolla", "aceite motor", "aceite motor", "TOY-OIL-001"),
        ("VIN023", "Toyota", 2020, "Corolla", "aceite motor", "aceite motor", "TOY-OIL-001"),
        ("VIN024", "Toyota", 2020, "Corolla", "aceite motor", "aceite motor", "TOY-OIL-001"),
        ("VIN025", "Toyota", 2020, "Corolla", "aceite motor", "aceite motor", "TOY-OIL-001"),
        ("VIN026", "Toyota", 2020, "Corolla", "aceite motor", "aceite motor", "TOY-OIL-001"),
        ("VIN027", "Toyota", 2021, "Corolla", "aceite motor", "aceite motor", "TOY-OIL-001"),
        ("VIN028", "Toyota", 2021, "Corolla", "aceite motor", "aceite motor", "TOY-OIL-001"),
        ("VIN029", "Toyota", 2021, "Corolla", "aceite motor", "aceite motor", "TOY-OIL-001"),
    ]
    
    # Insert test data
    for record in test_data:
        cursor.execute("""
            INSERT INTO processed_consolidado 
            (vin_number, maker, model, series, descripcion, normalized_descripcion, referencia)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, record)
    
    conn.commit()
    print(f"✅ Inserted {len(test_data)} test records")
    
    # Now populate year ranges
    print("Populating year ranges...")
    
    # Aggregate SKU data
    cursor.execute("""
        SELECT maker, series, descripcion, normalized_descripcion, referencia, 
               GROUP_CONCAT(model) as years, COUNT(*) as frequency
        FROM processed_consolidado 
        WHERE referencia IS NOT NULL AND referencia != '' AND referencia != 'UNKNOWN'
        GROUP BY maker, series, descripcion, referencia
    """)
    
    sku_data = cursor.fetchall()
    
    def detect_year_ranges(years):
        """Detect year ranges from a list of years."""
        if not years:
            return []
        
        years = sorted(set(years))
        
        if len(years) == 1:
            return [(years[0], years[0])]
        
        # For automotive parts, create one range from min to max year
        return [(min(years), max(years))]
    
    # Process SKU year ranges
    for row in sku_data:
        maker, series, descripcion, normalized_descripcion, referencia, years_str, frequency = row
        
        # Parse years
        years = []
        if years_str:
            for year_str in years_str.split(','):
                try:
                    years.append(int(year_str))
                except ValueError:
                    continue
        
        if not years:
            continue
        
        # Detect year ranges
        year_ranges = detect_year_ranges(years)
        
        for start_year, end_year in year_ranges:
            cursor.execute("""
                INSERT OR REPLACE INTO sku_year_ranges 
                (maker, series, descripcion, normalized_descripcion, referencia, start_year, end_year, frequency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (maker, series, descripcion, normalized_descripcion, referencia, start_year, end_year, frequency))
    
    # Aggregate VIN data
    cursor.execute("""
        SELECT maker, series, GROUP_CONCAT(model) as years, COUNT(*) as frequency
        FROM processed_consolidado 
        WHERE maker IS NOT NULL AND series IS NOT NULL
        GROUP BY maker, series
    """)
    
    vin_data = cursor.fetchall()
    
    # Process VIN year ranges
    for row in vin_data:
        maker, series, years_str, frequency = row
        
        # Parse years
        years = []
        if years_str:
            for year_str in years_str.split(','):
                try:
                    years.append(int(year_str))
                except ValueError:
                    continue
        
        if not years:
            continue
        
        # Detect year ranges
        year_ranges = detect_year_ranges(years)
        
        for start_year, end_year in year_ranges:
            cursor.execute("""
                INSERT OR REPLACE INTO vin_year_ranges 
                (maker, series, start_year, end_year, frequency)
                VALUES (?, ?, ?, ?, ?)
            """, (maker, series, start_year, end_year, frequency))
    
    conn.commit()
    
    # Show results
    cursor.execute("SELECT COUNT(*) FROM processed_consolidado")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM sku_year_ranges")
    sku_ranges = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM vin_year_ranges")
    vin_ranges = cursor.fetchone()[0]
    
    print(f"✅ Total records: {total_records}")
    print(f"✅ SKU year ranges: {sku_ranges}")
    print(f"✅ VIN year ranges: {vin_ranges}")
    
    # Show sample year ranges
    print(f"\n=== Sample SKU Year Ranges ===")
    cursor.execute("""
        SELECT maker, series, referencia, start_year, end_year, frequency 
        FROM sku_year_ranges 
        ORDER BY frequency DESC 
        LIMIT 10
    """)
    
    for row in cursor.fetchall():
        maker, series, referencia, start_year, end_year, frequency = row
        print(f"  {maker}/{series} - {referencia}: {start_year}-{end_year} (freq: {frequency})")
    
    conn.close()
    return True

if __name__ == "__main__":
    success = create_test_data()
    if success:
        print("\n🎉 Test data created successfully!")
    else:
        print("\n💥 Failed to create test data!")
