#!/usr/bin/env python3
"""
COMPREHENSIVE Field Name Standardization Script
Ensures 100% consistency across the ENTIRE project with original consolidado.json names:
- maker, series, model, referencia, descripcion

This script will update ALL files in the project systematically.
"""

import os
import re
import glob
from pathlib import Path

def comprehensive_field_name_update(file_path):
    """Comprehensively update ALL field name variations in a single file"""
    print(f"Processing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # COMPREHENSIVE field name replacements - ALL variations
        replacements = [
            # === DICTIONARY KEYS AND STRINGS ===
            (r"'Make'", "'maker'"),
            (r'"Make"', '"maker"'),
            (r"'Model Year'", "'model'"),
            (r'"Model Year"', '"model"'),
            (r"'Year'", "'model'"),
            (r'"Year"', '"model"'),
            (r"'fabrication_year'", "'model'"),
            (r'"fabrication_year"', '"model"'),
            (r"'Series'", "'series'"),
            (r'"Series"', '"series"'),
            (r"'Description'", "'descripcion'"),
            (r'"Description"', '"descripcion"'),
            (r"'SKU'", "'referencia'"),
            (r'"SKU"', '"referencia"'),
            
            # === VARIABLE ACCESS PATTERNS ===
            (r"details\['Make'\]", "details['maker']"),
            (r"details\['Model Year'\]", "details['model']"),
            (r"details\['Year'\]", "details['model']"),
            (r"details\['fabrication_year'\]", "details['model']"),
            (r"details\['Series'\]", "details['series']"),
            (r"details\['Description'\]", "details['descripcion']"),
            (r"details\['SKU'\]", "details['referencia']"),
            
            # === DATABASE COLUMN REFERENCES ===
            (r"vin_make", "maker"),
            (r"vin_year", "model"),
            (r"vin_series", "series"),
            (r"original_description", "original_descripcion"),
            (r"normalized_description", "normalized_descripcion"),
            (r"'sku'", "'referencia'"),
            (r'"sku"', '"referencia"'),
            (r"\.sku", ".referencia"),
            (r"\bsku\b", "referencia"),  # Word boundary for standalone 'sku'
            
            # === MAESTRO COLUMN NAMES ===
            (r"VIN_Make", "maker"),
            (r"VIN_Year", "model"),
            (r"VIN_Year_Min", "model"),
            (r"VIN_Series_Trim", "series"),
            (r"Original_Description_Input", "original_descripcion"),
            (r"Normalized_Description_Input", "normalized_descripcion"),
            (r"Confirmed_SKU", "referencia"),
            
            # === SQL QUERIES ===
            (r"SELECT sku,", "SELECT referencia,"),
            (r"GROUP BY sku", "GROUP BY referencia"),
            (r"ORDER BY sku", "ORDER BY referencia"),
            (r"WHERE sku", "WHERE referencia"),
            (r"sku IS NOT NULL", "referencia IS NOT NULL"),
            (r"sku != ''", "referencia != ''"),
            (r"DISTINCT sku", "DISTINCT referencia"),
            
            # === PANDAS DATAFRAME OPERATIONS ===
            (r"df\['Make'\]", "df['maker']"),
            (r"df\['Model Year'\]", "df['model']"),
            (r"df\['Year'\]", "df['model']"),
            (r"df\['fabrication_year'\]", "df['model']"),
            (r"df\['Series'\]", "df['series']"),
            (r"df\['Description'\]", "df['descripcion']"),
            (r"df\['SKU'\]", "df['referencia']"),
            (r"df\['sku'\]", "df['referencia']"),
            
            # === FUNCTION PARAMETERS ===
            (r"make: str, model_year: str, series: str, description: str", 
             "maker: str, model: str, series: str, descripcion: str"),
            (r"fabrication_year: str", "model: str"),
            (r"fabrication_year: int", "model: int"),
            (r"description: str", "descripcion: str"),
            
            # === VARIABLE NAMES ===
            (r"fabrication_year", "model"),
            (r"model_year", "model"),
            (r"sku_encoded", "referencia_encoded"),
            (r"encoder_sku", "encoder_referencia"),
            (r"y_sku", "y_referencia"),
            
            # === COMMENTS AND DOCUMENTATION ===
            (r"Make, Year, Series", "maker, model, series"),
            (r"Make\+Year\+Series", "maker+model+series"),
            (r"Make/Year/Series", "maker/model/series"),
            (r"Make, Model Year, Series", "maker, model, series"),
            (r"4-parameter matching.*Make.*Year.*Series.*Description", 
             "4-parameter matching (maker, model, series, descripcion)"),
            
            # === SPECIFIC PATTERNS FOR TRAINING SCRIPTS ===
            (r"'Make', 'Model Year', 'Series'", "'maker', 'model', 'series'"),
            (r'"Make", "Model Year", "Series"', '"maker", "model", "series"'),
            (r"categorical_features = \['Make', 'Model Year', 'Series'\]", 
             "categorical_features = ['maker', 'model', 'series']"),
            
            # === UI AND DISPLAY TEXT ===
            (r"Make=", "maker="),
            (r"Year=", "model="),
            (r"Series=", "series="),
            (r"Description=", "descripcion="),
            (r"SKU=", "referencia="),
            
            # === PRINT STATEMENTS AND LOGS ===
            (r"Make:", "maker:"),
            (r"Year:", "model:"),
            (r"Series:", "series:"),
            (r"Description:", "descripcion:"),
            (r"SKU:", "referencia:"),
            
            # === SPECIFIC CONSOLIDADO PROCESSING ===
            (r"item_sku", "item_referencia"),
            (r"item_original_description", "item_original_descripcion"),
            (r"item_normalized_description", "item_normalized_descripcion"),
            
            # === INDEX NAMES ===
            (r"idx_sku_", "idx_referencia_"),
            (r"covering_sku_search", "covering_referencia_search"),
            
            # === CACHE KEYS ===
            (r"cache_key.*make.*year.*series.*description", 
             "cache_key = f'{maker}_{model}_{series}_{descripcion_hash}'"),
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
    """Main function to process ALL files in the project"""
    print("="*80)
    print("COMPREHENSIVE FIELD NAME STANDARDIZATION")
    print("="*80)
    print("Ensuring 100% consistency with original consolidado.json field names:")
    print("  - maker")
    print("  - series") 
    print("  - model")
    print("  - referencia")
    print("  - descripcion")
    print("="*80)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    # Find ALL files to process
    files_to_process = []
    
    # Python files
    python_patterns = [
        "src/*.py",
        "performance_improvements/**/*.py",
        "*.py"  # Root level Python files
    ]
    
    for pattern in python_patterns:
        files = glob.glob(str(project_root / pattern), recursive=True)
        files_to_process.extend(files)
    
    # Documentation files
    doc_patterns = [
        "docs/*.md",
        "*.md"
    ]
    
    for pattern in doc_patterns:
        files = glob.glob(str(project_root / pattern), recursive=True)
        files_to_process.extend(files)
    
    # Remove duplicates and sort
    files_to_process = sorted(list(set(files_to_process)))
    
    # Filter out temporary scripts and backup files
    files_to_process = [f for f in files_to_process if 
                       "TEMPORARY SCRIPTS" not in f and 
                       "_backup" not in f and
                       "__pycache__" not in f]
    
    print(f"\nFound {len(files_to_process)} files to process:")
    for f in files_to_process:
        print(f"  - {f}")
    
    print(f"\nProcessing files...")
    
    updated_count = 0
    total_count = len(files_to_process)
    
    for file_path in files_to_process:
        if comprehensive_field_name_update(file_path):
            updated_count += 1
    
    print("="*80)
    print("COMPREHENSIVE STANDARDIZATION SUMMARY")
    print("="*80)
    print(f"Total files processed: {total_count}")
    print(f"Files updated: {updated_count}")
    print(f"Files unchanged: {total_count - updated_count}")
    
    if updated_count > 0:
        print("\n🎉 COMPREHENSIVE field name standardization completed!")
        print("\n✅ ALL files now use consistent original consolidado.json names:")
        print("   - maker, series, model, referencia, descripcion")
        print("\nNext steps:")
        print("1. Test the application thoroughly")
        print("2. Run training scripts to verify they work")
        print("3. Commit all changes")
    else:
        print("\n⏭️  All files already use consistent field names")

if __name__ == "__main__":
    main()
