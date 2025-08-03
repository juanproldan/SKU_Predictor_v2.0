#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create Make-Series-Year-Ranges Report

This script analyzes the processed_consolidado database and creates an Excel report
with maker, series, year ranges, and frequency information for review.
"""

import os
import sys
import sqlite3
import pandas as pd
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_make_series_report():
    """Create comprehensive make-series-year-ranges report."""
    
    try:
        from unified_consolidado_processor import get_base_path
        
        print("📊 Creating Make-Series-Year-Ranges Report")
        print("=" * 50)
        
        # Get paths
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        output_path = os.path.join(base_path, "Source_Files", 
                                 "make_series_year_ranges NO ES FUENTE, ES PARA REVISAR CON JUAN MARTIN.xlsx")
        
        print(f"📂 Database: {db_path}")
        print(f"📂 Output: {output_path}")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔍 Analyzing database...")
        
        # Query to get maker, series, year ranges, and frequency
        query = """
        SELECT
            maker,
            series,
            model,
            COUNT(*) as frequency,
            COUNT(DISTINCT referencia) as unique_skus,
            COUNT(DISTINCT descripcion) as unique_descriptions
        FROM processed_consolidado
        WHERE maker IS NOT NULL
        AND series IS NOT NULL
        AND maker != ''
        AND series != ''
        AND model IS NOT NULL
        AND model != ''
        GROUP BY maker, series, model
        ORDER BY maker, series, model
        """
        
        cursor.execute(query)
        results = cursor.fetchall()

        print(f"✅ Found {len(results)} maker-series-model combinations")

        # Create DataFrame with raw data
        raw_df = pd.DataFrame(results, columns=[
            'maker', 'series', 'model', 'frequency', 'unique_skus', 'unique_descriptions'
        ])

        print(f"🔍 Processing year data...")

        # Filter for numeric years and convert
        def is_numeric_year(year_str):
            try:
                year = int(str(year_str))
                return 1990 <= year <= 2030  # Reasonable year range
            except:
                return False

        # Filter for valid years
        valid_years = raw_df[raw_df['model'].apply(is_numeric_year)].copy()
        valid_years['year'] = valid_years['model'].astype(int)

        print(f"✅ Found {len(valid_years)} records with valid years")

        # Group by maker and series to get year ranges
        grouped = valid_years.groupby(['maker', 'series']).agg({
            'year': ['min', 'max'],
            'frequency': 'sum',
            'unique_skus': 'sum',
            'unique_descriptions': 'sum'
        }).reset_index()

        # Flatten column names
        grouped.columns = ['maker', 'series', 'start_year', 'end_year',
                          'frequency', 'unique_skus', 'unique_descriptions']

        # Calculate year span
        grouped['year_span'] = grouped['end_year'] - grouped['start_year'] + 1

        # Sort by frequency
        df = grouped.sort_values(['maker', 'frequency'], ascending=[True, False])
        
        # Add some statistics
        print(f"📈 Statistics:")
        print(f"   Total makers: {df['maker'].nunique()}")
        print(f"   Total series: {df['series'].nunique()}")
        print(f"   Total records: {df['frequency'].sum():,}")
        print(f"   Year range: {df['start_year'].min()}-{df['end_year'].max()}")
        
        # Get top makers by frequency
        top_makers = df.groupby('maker')['frequency'].sum().sort_values(ascending=False).head(10)
        print(f"\n🏆 Top 10 Makers by Frequency:")
        for maker, freq in top_makers.items():
            series_count = df[df['maker'] == maker]['series'].nunique()
            print(f"   {maker}: {freq:,} records, {series_count} series")
        
        # Create summary sheet data
        summary_data = []
        
        # Overall statistics
        summary_data.append(['Total Makers', df['maker'].nunique()])
        summary_data.append(['Total Series', df['series'].nunique()])
        summary_data.append(['Total Records', df['frequency'].sum()])
        summary_data.append(['Total Unique SKUs', df['unique_skus'].sum()])
        summary_data.append(['Year Range', f"{df['start_year'].min()}-{df['end_year'].max()}"])
        summary_data.append([''])  # Empty row
        
        # Top makers
        summary_data.append(['Top Makers by Frequency', ''])
        for maker, freq in top_makers.head(5).items():
            series_count = df[df['maker'] == maker]['series'].nunique()
            summary_data.append([maker, f"{freq:,} records, {series_count} series"])
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        
        # Create maker summary
        maker_summary = df.groupby('maker').agg({
            'series': 'nunique',
            'frequency': 'sum',
            'unique_skus': 'sum',
            'start_year': 'min',
            'end_year': 'max'
        }).reset_index()
        
        maker_summary.columns = ['maker', 'series_count', 'total_frequency', 
                               'total_unique_skus', 'earliest_year', 'latest_year']
        maker_summary['year_span'] = maker_summary['latest_year'] - maker_summary['earliest_year'] + 1
        maker_summary = maker_summary.sort_values('total_frequency', ascending=False)
        
        # Write to Excel with multiple sheets
        print("📝 Writing Excel file...")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Make_Series_Details', index=False)
            
            # Summary sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Maker summary sheet
            maker_summary.to_excel(writer, sheet_name='Maker_Summary', index=False)
            
            # Top series by maker (first 5 makers)
            for i, (maker, maker_data) in enumerate(df.groupby('maker')):
                if i >= 5:  # Limit to first 5 makers to avoid too many sheets
                    break
                    
                maker_series = maker_data.sort_values('frequency', ascending=False)
                sheet_name = f"{maker[:20]}_Series"  # Limit sheet name length
                maker_series.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Excel file created: {output_path}")
        
        # Show sample data
        print(f"\n📋 Sample Data (Top 10 by frequency):")
        sample = df.head(10)[['maker', 'series', 'start_year', 'end_year', 'frequency']]
        for _, row in sample.iterrows():
            print(f"   {row['maker']} | {row['series'][:30]}... | {row['start_year']}-{row['end_year']} | {row['frequency']:,}")
        
        conn.close()
        
        # Check if file should be tracked by git
        print(f"\n📁 Git Tracking:")
        if os.path.exists('.git'):
            print(f"   ✅ Git repository detected")
            print(f"   📄 File created: {os.path.basename(output_path)}")
            print(f"   💡 To track this file, run:")
            print(f"      git add \"{os.path.basename(output_path)}\"")
            print(f"      git commit -m \"Add make-series-year-ranges analysis report\"")
        else:
            print(f"   ⚠️ No git repository found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating report: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_make_series_report()
    
    if success:
        print("\n🎉 Make-Series-Year-Ranges report created successfully!")
    else:
        print("\n💥 Report creation failed!")
