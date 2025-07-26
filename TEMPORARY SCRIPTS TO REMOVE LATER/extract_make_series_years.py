"""
Extract Make-Series Year Ranges from processed_consolidado.db

This script analyzes the processed_consolidado database to create a summary table
showing the year range (start year to end year) for each unique Make-Series combination.

Output: Excel file with columns: Make, Series, Start Year, End Year
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

def extract_make_series_years():
    """
    Extract Make-Series combinations with their year ranges from processed_consolidado.db
    """
    
    # Database path
    db_path = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\Source_Files\processed_consolidado.db"
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return
    
    print(f"📊 Connecting to database: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check table structure first
        cursor.execute("PRAGMA table_info(processed_consolidado)")
        columns = cursor.fetchall()
        print(f"📋 Database columns found:")
        for col in columns:
            print(f"   - {col[1]} ({col[2]})")
        
        # Query to get Make-Series year ranges
        print(f"\n🔍 Extracting Make-Series year ranges...")
        
        query = """
        SELECT 
            vin_make as Make,
            vin_series as Series,
            MIN(vin_year) as Start_Year,
            MAX(vin_year) as End_Year,
            COUNT(*) as Total_Records
        FROM processed_consolidado 
        WHERE vin_make IS NOT NULL 
            AND vin_make != '' 
            AND vin_series IS NOT NULL 
            AND vin_series != ''
            AND vin_year IS NOT NULL
            AND vin_year != ''
        GROUP BY vin_make, vin_series
        ORDER BY vin_make, vin_series
        """
        
        # Execute query
        cursor.execute(query)
        results = cursor.fetchall()
        
        print(f"✅ Found {len(results)} unique Make-Series combinations")
        
        # Convert to DataFrame
        df = pd.DataFrame(results, columns=['Make', 'Series', 'Start_Year', 'End_Year', 'Total_Records'])
        
        # Clean up data types
        df['Start_Year'] = pd.to_numeric(df['Start_Year'], errors='coerce').astype('Int64')
        df['End_Year'] = pd.to_numeric(df['End_Year'], errors='coerce').astype('Int64')
        df['Total_Records'] = pd.to_numeric(df['Total_Records'], errors='coerce').astype('Int64')
        
        # Remove rows with invalid years
        df = df.dropna(subset=['Start_Year', 'End_Year'])
        
        # Show preview
        print(f"\n📋 Preview of results:")
        print(df.head(10).to_string(index=False))
        
        # Show summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"   - Total Make-Series combinations: {len(df)}")
        print(f"   - Unique Makes: {df['Make'].nunique()}")
        print(f"   - Year range: {df['Start_Year'].min()} - {df['End_Year'].max()}")
        print(f"   - Total records analyzed: {df['Total_Records'].sum():,}")
        
        # Save to Excel
        output_path = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\TEMPORARY SCRIPTS TO REMOVE LATER\Make_Series_Year_Ranges.xlsx"
        
        print(f"\n💾 Saving to Excel: {output_path}")
        
        # Create Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main summary sheet
            df_summary = df[['Make', 'Series', 'Start_Year', 'End_Year']].copy()
            df_summary.to_excel(writer, sheet_name='Make_Series_Years', index=False)
            
            # Detailed sheet with record counts
            df.to_excel(writer, sheet_name='Detailed_with_Counts', index=False)
            
            # Summary by Make
            make_summary = df.groupby('Make').agg({
                'Series': 'count',
                'Start_Year': 'min',
                'End_Year': 'max',
                'Total_Records': 'sum'
            }).rename(columns={'Series': 'Series_Count'}).reset_index()
            make_summary.to_excel(writer, sheet_name='Summary_by_Make', index=False)
        
        print(f"✅ Excel file saved successfully!")
        print(f"📁 Location: {output_path}")
        
        # Show top makes by series count
        print(f"\n🏆 Top Makes by Series Count:")
        top_makes = make_summary.nlargest(10, 'Series_Count')[['Make', 'Series_Count', 'Total_Records']]
        print(top_makes.to_string(index=False))
        
        conn.close()
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'conn' in locals():
            conn.close()
        return None

def main():
    """Main function"""
    print("🚀 Make-Series Year Range Extractor")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    output_dir = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\TEMPORARY SCRIPTS TO REMOVE LATER"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    result = extract_make_series_years()
    
    if result:
        print(f"\n🎉 Process completed successfully!")
        print(f"📊 Excel file ready for analysis: {result}")
    else:
        print(f"\n❌ Process failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
