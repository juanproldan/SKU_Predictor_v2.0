#!/usr/bin/env python3
"""
Script to standardize field names throughout the project to use original consolidado.json names:
- Make -> maker
- Year/Model Year -> fabrication_year  
- Series -> series (already correct)
- Description -> descripcion
- SKU -> referencia

This script will update all Python files in the project.
"""

import os
import re
import glob
from pathlib import Path

def update_field_names_in_file(file_path):
    """Update field names in a single file"""
    print(f"Processing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Field name mappings - be careful with order and context
        replacements = [
            # Dictionary key replacements (most specific first)
            (r"'Make'", "'maker'"),
            (r'"Make"', '"maker"'),
            (r"'Model Year'", "'fabrication_year'"),
            (r'"Model Year"', '"fabrication_year"'),
            (r"'Year'", "'fabrication_year'"),
            (r'"Year"', '"fabrication_year"'),
            (r"'Description'", "'descripcion'"),
            (r'"Description"', '"descripcion"'),
            (r"'SKU'", "'referencia'"),
            (r'"SKU"', '"referencia"'),
            
            # Variable and field access patterns
            (r"details\['Make'\]", "details['maker']"),
            (r"details\['Model Year'\]", "details['fabrication_year']"),
            (r"details\['Year'\]", "details['fabrication_year']"),
            (r"details\['Series'\]", "details['series']"),
            (r"details\['Description'\]", "details['descripcion']"),
            (r"details\['SKU'\]", "details['referencia']"),
            
            # Database column references
            (r"vin_make", "maker"),
            (r"vin_year", "fabrication_year"),
            (r"vin_series", "series"),
            (r"original_description", "original_descripcion"),
            (r"normalized_description", "normalized_descripcion"),
            (r"'sku'", "'referencia'"),
            (r'"sku"', '"referencia"'),
            
            # SQL column names in queries
            (r"SELECT sku,", "SELECT referencia,"),
            (r"GROUP BY sku", "GROUP BY referencia"),
            (r"ORDER BY sku", "ORDER BY referencia"),
            
            # Specific patterns for training scripts
            (r"'Make', 'Model Year', 'Series'", "'maker', 'fabrication_year', 'series'"),
            (r'"Make", "Model Year", "Series"', '"maker", "fabrication_year", "series"'),
            
            # Comment and string updates
            (r"Make, Year, Series", "maker, fabrication_year, series"),
            (r"Make\+Year\+Series", "maker+fabrication_year+series"),
            (r"Make/Year/Series", "maker/fabrication_year/series"),
            
            # Specific function parameter names
            (r"make: str, model_year: str, series: str, description: str", 
             "maker: str, fabrication_year: str, series: str, descripcion: str"),
            
            # UI and display text
            (r"Make=", "maker="),
            (r"Year=", "fabrication_year="),
            (r"Series=", "series="),
            
            # Encoder and model variable names (be careful not to break existing working code)
            (r"sku_encoded", "referencia_encoded"),
            (r"encoder_sku", "encoder_referencia"),
            
            # Pandas DataFrame column operations
            (r"df\['Make'\]", "df['maker']"),
            (r"df\['Model Year'\]", "df['fabrication_year']"),
            (r"df\['Year'\]", "df['fabrication_year']"),
            (r"df\['Series'\]", "df['series']"),
            (r"df\['Description'\]", "df['descripcion']"),
            (r"df\['SKU'\]", "df['referencia']"),
            (r"df\['sku'\]", "df['referencia']"),
        ]
        
        # Apply replacements
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Updated: {file_path}")
            return True
        else:
            print(f"  ⏭️  No changes: {file_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all Python files"""
    print("="*80)
    print("FIELD NAME STANDARDIZATION SCRIPT")
    print("="*80)
    print("Converting to original consolidado.json field names:")
    print("  Make -> maker")
    print("  Year/Model Year -> fabrication_year")
    print("  Series -> series (no change)")
    print("  Description -> descripcion")
    print("  SKU -> referencia")
    print("="*80)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    # Find all Python files to process
    python_files = []
    
    # Main source files
    src_files = glob.glob(str(project_root / "src" / "*.py"))
    python_files.extend(src_files)
    
    # Performance improvement files
    perf_files = glob.glob(str(project_root / "performance_improvements" / "**" / "*.py"), recursive=True)
    python_files.extend(perf_files)
    
    # Validation files
    validation_files = glob.glob(str(project_root / "performance_improvements" / "validation" / "*.py"))
    python_files.extend(validation_files)
    
    # Cache files
    cache_files = glob.glob(str(project_root / "performance_improvements" / "cache" / "*.py"))
    python_files.extend(cache_files)
    
    # Remove duplicates and sort
    python_files = sorted(list(set(python_files)))
    
    print(f"\nFound {len(python_files)} Python files to process:")
    for f in python_files:
        print(f"  - {f}")
    
    print(f"\nProcessing files...")
    
    updated_count = 0
    total_count = len(python_files)
    
    for file_path in python_files:
        if update_field_names_in_file(file_path):
            updated_count += 1
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files processed: {total_count}")
    print(f"Files updated: {updated_count}")
    print(f"Files unchanged: {total_count - updated_count}")
    
    if updated_count > 0:
        print("\n✅ Field name standardization completed successfully!")
        print("\nNext steps:")
        print("1. Test the application to ensure all changes work correctly")
        print("2. Update any remaining database schemas manually if needed")
        print("3. Update documentation and comments as needed")
    else:
        print("\n⏭️  No files needed updating - field names may already be standardized")

if __name__ == "__main__":
    main()
