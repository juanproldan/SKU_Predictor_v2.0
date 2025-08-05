#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create Enhanced Make-Series-Year-Ranges Report with VIN Patterns

This script analyzes the processed_consolidado database and creates an Excel report
with maker, series, year ranges, frequency information, and VIN patterns for review.

ENHANCEMENTS:
- Added VIN patterns (first 8 characters: WMI + VDS)
- Added sample VINs for each series
- Better formatting and organization
"""

import os
import sys
import sqlite3
import pandas as pd
import xlsxwriter
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_enhanced_make_series_report():
    """Create comprehensive make-series-year-ranges report with VIN patterns."""
    
    try:
        from unified_consolidado_processor import get_base_path
        
        print("📊 Creating Enhanced Make-Series-Year-Ranges Report with VIN Patterns")
        print("=" * 70)
        
        # Get paths
        base_path = get_base_path()
        db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        # Store output in same folder as consolidado.json
        output_path = os.path.join(base_path, "Source_Files",
                                 "make_series_year_ranges NO ES FUENTE, ES PARA REVISAR CON JUAN MARTIN.xlsx")
        
        print(f"📂 Database: {db_path}")
        print(f"📂 Output: {output_path}")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔍 Analyzing database with VIN patterns...")
        
        # Enhanced query to get maker, series, year ranges, frequency, and VIN patterns
        query = """
        SELECT
            maker,
            series,
            model,
            COUNT(*) as frequency,
            COUNT(DISTINCT referencia) as unique_skus,
            COUNT(DISTINCT descripcion) as unique_descriptions,
            GROUP_CONCAT(DISTINCT SUBSTR(vin_number, 1, 8)) as vin_patterns,
            GROUP_CONCAT(DISTINCT vin_number) as sample_vins
        FROM processed_consolidado
        WHERE maker IS NOT NULL
        AND series IS NOT NULL
        AND model IS NOT NULL
        AND referencia IS NOT NULL
        GROUP BY maker, series, model
        ORDER BY maker, series, model
        """
        
        cursor.execute(query)
        raw_data = cursor.fetchall()
        
        print(f"✅ Found {len(raw_data)} maker-series-year combinations")
        
        # Convert to DataFrame for easier processing
        columns = ['maker', 'series', 'model', 'frequency', 'unique_skus', 'unique_descriptions', 'vin_patterns', 'sample_vins']
        raw_df = pd.DataFrame(raw_data, columns=columns)
        
        # Function to check if model is a valid year
        def is_numeric_year(model):
            try:
                year = int(model)
                return 1990 <= year <= 2030  # Reasonable year range
            except:
                return False

        # Filter for valid years
        valid_years = raw_df[raw_df['model'].apply(is_numeric_year)].copy()
        valid_years['year'] = valid_years['model'].astype(int)

        print(f"✅ Found {len(valid_years)} records with valid years")

        # Group by maker and series to get year ranges with VIN patterns
        grouped_data = []
        
        for (maker, series), group in valid_years.groupby(['maker', 'series']):
            start_year = group['year'].min()
            end_year = group['year'].max()
            total_frequency = group['frequency'].sum()
            total_unique_skus = group['unique_skus'].sum()
            total_unique_descriptions = group['unique_descriptions'].sum()
            year_span = end_year - start_year + 1
            
            # Process VIN patterns
            all_patterns = []
            all_sample_vins = []
            
            for _, row in group.iterrows():
                if row['vin_patterns'] and pd.notna(row['vin_patterns']):
                    patterns = [p.strip() for p in str(row['vin_patterns']).split(',') if p.strip()]
                    all_patterns.extend(patterns)
                
                if row['sample_vins'] and pd.notna(row['sample_vins']):
                    samples = [v.strip() for v in str(row['sample_vins']).split(',') if v.strip() and len(v.strip()) == 17]
                    all_sample_vins.extend(samples)
            
            # Get unique patterns and samples
            unique_patterns = list(set(all_patterns))[:3]  # Top 3 patterns
            unique_samples = list(set(all_sample_vins))[:2]  # Top 2 sample VINs
            
            vin_patterns_display = ', '.join(unique_patterns) if unique_patterns else 'No VINs'
            sample_vins_display = ', '.join(unique_samples) if unique_samples else 'No VINs'
            
            grouped_data.append({
                'maker': maker,
                'series': series,
                'start_year': start_year,
                'end_year': end_year,
                'frequency': total_frequency,
                'unique_skus': total_unique_skus,
                'unique_descriptions': total_unique_descriptions,
                'year_span': year_span,
                'vin_patterns': vin_patterns_display,
                'sample_vins': sample_vins_display
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(grouped_data)
        
        # Sort by frequency
        df = df.sort_values(['maker', 'frequency'], ascending=[True, False])
        
        print(f"📈 Enhanced Statistics:")
        print(f"   Total makers: {df['maker'].nunique()}")
        print(f"   Total series: {df['series'].nunique()}")
        print(f"   Total records: {df['frequency'].sum():,}")
        print(f"   Year range: {df['start_year'].min()}-{df['end_year'].max()}")
        print(f"   Series with VIN patterns: {len(df[df['vin_patterns'] != 'No VINs'])}")
        
        # Create Excel workbook
        workbook = xlsxwriter.Workbook(output_path)
        
        # Create enhanced detailed sheet
        create_enhanced_details_sheet(workbook, df)
        
        # Create summary sheet
        create_enhanced_summary_sheet(workbook, df)
        
        # Create maker breakdown sheets
        create_maker_breakdown_sheets(workbook, df)

        # Create VIN patterns analysis sheet
        create_vin_patterns_analysis_sheet(workbook, conn)

        # Close workbook
        workbook.close()
        conn.close()
        
        print(f"\n🎉 Enhanced report created successfully!")
        print(f"📁 File: {output_path}")
        print(f"📊 Contains {len(df)} maker-series combinations with VIN patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating enhanced report: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_enhanced_details_sheet(workbook, df):
    """Create enhanced detailed sheet with VIN patterns."""
    
    print("📋 Creating enhanced Make_Series_Details sheet...")
    
    # Create worksheet
    worksheet = workbook.add_worksheet('Make_Series_Details')
    
    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'border': 1,
        'align': 'center'
    })
    
    data_format = workbook.add_format({
        'border': 1,
        'align': 'left'
    })
    
    number_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'num_format': '#,##0'
    })
    
    vin_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'font_name': 'Courier New',
        'font_size': 9
    })
    
    # Write headers (ENHANCED with VIN patterns)
    headers = ['maker', 'series', 'start_year', 'end_year', 'frequency', 'unique_skus', 'unique_descriptions', 'year_span', 'vin_patterns', 'sample_vins']
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
    
    # Write data
    for row, (_, record) in enumerate(df.iterrows(), 1):
        worksheet.write(row, 0, record['maker'], data_format)
        worksheet.write(row, 1, record['series'], data_format)
        worksheet.write(row, 2, record['start_year'], number_format)
        worksheet.write(row, 3, record['end_year'], number_format)
        worksheet.write(row, 4, record['frequency'], number_format)
        worksheet.write(row, 5, record['unique_skus'], number_format)
        worksheet.write(row, 6, record['unique_descriptions'], number_format)
        worksheet.write(row, 7, record['year_span'], number_format)
        worksheet.write(row, 8, record['vin_patterns'], vin_format)
        worksheet.write(row, 9, record['sample_vins'], vin_format)
    
    # Auto-adjust column widths
    worksheet.set_column('A:A', 15)  # maker
    worksheet.set_column('B:B', 40)  # series
    worksheet.set_column('C:H', 12)  # numbers
    worksheet.set_column('I:I', 30)  # vin_patterns
    worksheet.set_column('J:J', 40)  # sample_vins
    
    print(f"✅ Enhanced Make_Series_Details sheet created with {len(df)} records and VIN patterns")

def create_enhanced_summary_sheet(workbook, df):
    """Create enhanced summary sheet."""
    
    print("📋 Creating enhanced Summary sheet...")
    
    worksheet = workbook.add_worksheet('Summary')
    
    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#70AD47',
        'font_color': 'white',
        'border': 1,
        'align': 'center'
    })
    
    data_format = workbook.add_format({
        'border': 1,
        'align': 'left'
    })
    
    # Overall statistics
    summary_data = [
        ['Total Makers', df['maker'].nunique()],
        ['Total Series', df['series'].nunique()],
        ['Total Records', df['frequency'].sum()],
        ['Total Unique SKUs', df['unique_skus'].sum()],
        ['Year Range', f"{df['start_year'].min()}-{df['end_year'].max()}"],
        ['Series with VIN Patterns', len(df[df['vin_patterns'] != 'No VINs'])],
        [''],  # Empty row
        ['Top Makers by Frequency', '']
    ]
    
    # Top makers
    top_makers = df.groupby('maker')['frequency'].sum().sort_values(ascending=False).head(10)
    for maker, freq in top_makers.items():
        series_count = df[df['maker'] == maker]['series'].nunique()
        vin_series_count = len(df[(df['maker'] == maker) & (df['vin_patterns'] != 'No VINs')])
        summary_data.append([maker, f"{freq:,} records, {series_count} series, {vin_series_count} with VINs"])
    
    # Write summary data
    worksheet.write(0, 0, 'Metric', header_format)
    worksheet.write(0, 1, 'Value', header_format)

    for row, data_item in enumerate(summary_data, 1):
        if len(data_item) == 2:
            metric, value = data_item
            worksheet.write(row, 0, metric, data_format)
            worksheet.write(row, 1, str(value), data_format)
        else:
            # Handle single item (empty row)
            worksheet.write(row, 0, str(data_item[0]) if data_item else '', data_format)
            worksheet.write(row, 1, '', data_format)
    
    # Auto-adjust column widths
    worksheet.set_column('A:A', 25)
    worksheet.set_column('B:B', 50)
    
    print(f"✅ Enhanced Summary sheet created")

def create_maker_breakdown_sheets(workbook, df):
    """Create individual sheets for top makers."""
    
    print("📋 Creating maker breakdown sheets...")
    
    # Get top 5 makers by frequency
    top_makers = df.groupby('maker')['frequency'].sum().sort_values(ascending=False).head(5)

    created_sheets = set()
    for i, maker in enumerate(top_makers.index):
        maker_data = df[df['maker'] == maker].sort_values('frequency', ascending=False)

        # Create unique sheet name (Excel sheet names can't exceed 31 characters)
        base_name = f"{maker}_Series"[:25]  # Leave room for number
        sheet_name = base_name
        counter = 1
        while sheet_name.upper() in created_sheets:
            sheet_name = f"{base_name}_{counter}"[:31]
            counter += 1

        created_sheets.add(sheet_name.upper())
        worksheet = workbook.add_worksheet(sheet_name)
        
        # Use same formats as details sheet
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1,
            'align': 'center'
        })
        
        data_format = workbook.add_format({'border': 1, 'align': 'left'})
        number_format = workbook.add_format({'border': 1, 'align': 'center', 'num_format': '#,##0'})
        vin_format = workbook.add_format({'border': 1, 'align': 'center', 'font_name': 'Courier New', 'font_size': 9})
        
        # Write headers
        headers = ['series', 'start_year', 'end_year', 'frequency', 'unique_skus', 'year_span', 'vin_patterns', 'sample_vins']
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Write data
        for row, (_, record) in enumerate(maker_data.iterrows(), 1):
            worksheet.write(row, 0, record['series'], data_format)
            worksheet.write(row, 1, record['start_year'], number_format)
            worksheet.write(row, 2, record['end_year'], number_format)
            worksheet.write(row, 3, record['frequency'], number_format)
            worksheet.write(row, 4, record['unique_skus'], number_format)
            worksheet.write(row, 5, record['year_span'], number_format)
            worksheet.write(row, 6, record['vin_patterns'], vin_format)
            worksheet.write(row, 7, record['sample_vins'], vin_format)
        
        # Auto-adjust column widths
        worksheet.set_column('A:A', 40)  # series
        worksheet.set_column('B:F', 12)  # numbers
        worksheet.set_column('G:G', 30)  # vin_patterns
        worksheet.set_column('H:H', 40)  # sample_vins
    
    print(f"✅ Created breakdown sheets for {len(top_makers)} top makers")

def create_vin_patterns_analysis_sheet(workbook, conn):
    """Create VIN patterns analysis sheet showing patterns with their maker-series combinations."""

    print("📋 Creating VIN Patterns Analysis sheet...")

    # Create worksheet
    worksheet = workbook.add_worksheet('VIN_Patterns_Analysis')

    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#E67E22',  # Orange color for VIN patterns
        'font_color': 'white',
        'border': 1,
        'align': 'center'
    })

    data_format = workbook.add_format({
        'border': 1,
        'align': 'left'
    })

    number_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'num_format': '#,##0'
    })

    vin_pattern_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'font_name': 'Courier New',
        'font_size': 11,
        'bold': True
    })

    # Write headers exactly as requested
    headers = ['vin_pattern', 'maker', 'series', 'start_year', 'end_year', 'frequency', 'unique_skus', 'unique_descriptions', 'year_span']
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)

    # Query to get VIN patterns with their maker-series combinations
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            SUBSTR(vin_number, 1, 8) as vin_pattern,
            maker,
            series,
            MIN(model) as start_year,
            MAX(model) as end_year,
            COUNT(*) as frequency,
            COUNT(DISTINCT referencia) as unique_skus,
            COUNT(DISTINCT descripcion) as unique_descriptions,
            (MAX(model) - MIN(model) + 1) as year_span
        FROM processed_consolidado
        WHERE maker IS NOT NULL
        AND series IS NOT NULL
        AND model IS NOT NULL
        AND referencia IS NOT NULL
        AND vin_number IS NOT NULL
        AND LENGTH(vin_number) = 17
        AND vin_number != '00000000000000000'
        AND CAST(model AS INTEGER) BETWEEN 1990 AND 2030
        GROUP BY SUBSTR(vin_number, 1, 8), maker, series
        HAVING COUNT(*) >= 3
        ORDER BY vin_pattern, frequency DESC, maker, series
    """)

    # Write data
    row = 1
    for record in cursor.fetchall():
        vin_pattern, maker, series, start_year, end_year, frequency, unique_skus, unique_descriptions, year_span = record

        worksheet.write(row, 0, vin_pattern, vin_pattern_format)
        worksheet.write(row, 1, maker, data_format)
        worksheet.write(row, 2, series, data_format)
        worksheet.write(row, 3, start_year, number_format)
        worksheet.write(row, 4, end_year, number_format)
        worksheet.write(row, 5, frequency, number_format)
        worksheet.write(row, 6, unique_skus, number_format)
        worksheet.write(row, 7, unique_descriptions, number_format)
        worksheet.write(row, 8, year_span, number_format)

        row += 1

    # Auto-adjust column widths
    worksheet.set_column('A:A', 12)  # vin_pattern
    worksheet.set_column('B:B', 15)  # maker
    worksheet.set_column('C:C', 40)  # series
    worksheet.set_column('D:I', 12)  # numbers

    # Add filter to the header row
    worksheet.autofilter(0, 0, row-1, len(headers)-1)

    print(f"✅ VIN Patterns Analysis sheet created with {row-1} pattern-maker-series combinations")

    # Print some statistics
    cursor.execute("""
        SELECT
            COUNT(DISTINCT SUBSTR(vin_number, 1, 8)) as unique_patterns,
            COUNT(DISTINCT maker) as unique_makers,
            COUNT(DISTINCT series) as unique_series
        FROM processed_consolidado
        WHERE maker IS NOT NULL
        AND series IS NOT NULL
        AND model IS NOT NULL
        AND referencia IS NOT NULL
        AND vin_number IS NOT NULL
        AND LENGTH(vin_number) = 17
        AND vin_number != '00000000000000000'
        AND CAST(model AS INTEGER) BETWEEN 1990 AND 2030
    """)

    stats = cursor.fetchone()
    if stats:
        unique_patterns, unique_makers, unique_series = stats
        print(f"   📊 Statistics: {unique_patterns} unique VIN patterns, {unique_makers} makers, {unique_series} series")

if __name__ == "__main__":
    create_enhanced_make_series_report()
