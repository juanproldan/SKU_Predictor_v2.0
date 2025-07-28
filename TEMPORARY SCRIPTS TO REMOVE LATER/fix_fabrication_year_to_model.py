#!/usr/bin/env python3
"""
Script to fix fabrication_year -> model throughout the project.
The original consolidado.json uses 'model' field, not 'fabrication_year'.
"""

import os
import re
import glob
from pathlib import Path

def update_fabrication_year_to_model(file_path):
    """Update fabrication_year to model in a single file"""
    print(f"Processing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Field name replacements - fabrication_year -> model
        replacements = [
            # Dictionary key replacements
            (r"'fabrication_year'", "'model'"),
            (r'"fabrication_year"', '"model"'),
            
            # Variable and field access patterns
            (r"details\['fabrication_year'\]", "details['model']"),
            (r"fabrication_year", "model"),
            
            # SQL column names
            (r"fabrication_year INTEGER", "model INTEGER"),
            (r"fabrication_year TEXT", "model TEXT"),
            
            # Function parameters
            (r"fabrication_year: str", "model: str"),
            (r"fabrication_year: int", "model: int"),
            
            # Comments and documentation
            (r"fabrication_year", "model"),
            
            # Pandas DataFrame operations
            (r"df\['fabrication_year'\]", "df['model']"),
            
            # Database index names
            (r"maker, fabrication_year, series", "maker, model, series"),
            
            # Print statements and logs
            (r"fabrication_year=", "model="),
            (r"Year=", "model="),
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
    print("FABRICATION_YEAR -> MODEL CORRECTION SCRIPT")
    print("="*80)
    print("Converting fabrication_year to model to match original consolidado.json")
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
    
    # Remove duplicates and sort
    python_files = sorted(list(set(python_files)))
    
    print(f"\nFound {len(python_files)} Python files to process:")
    for f in python_files:
        print(f"  - {f}")
    
    print(f"\nProcessing files...")
    
    updated_count = 0
    total_count = len(python_files)
    
    for file_path in python_files:
        if update_fabrication_year_to_model(file_path):
            updated_count += 1
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files processed: {total_count}")
    print(f"Files updated: {updated_count}")
    print(f"Files unchanged: {total_count - updated_count}")
    
    if updated_count > 0:
        print("\n✅ fabrication_year -> model correction completed successfully!")
        print("\nNext steps:")
        print("1. Update PRD documentation")
        print("2. Test the application")
        print("3. Commit the changes")
    else:
        print("\n⏭️  No files needed updating")

if __name__ == "__main__":
    main()
