#!/usr/bin/env python3
"""
FINAL CONSISTENCY VALIDATION
Comprehensive check that ALL files use the correct original consolidado.json field names:
- maker, series, model, referencia, descripcion

This script will scan ALL files and report any inconsistencies.
"""

import os
import re
import glob
from pathlib import Path

def check_file_consistency(file_path):
    """Check a single file for field name consistency"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # OLD field names that should NOT exist anymore
        old_patterns = [
            # Old field names
            r"'Make'",
            r'"Make"',
            r"'Model Year'", 
            r'"Model Year"',
            r"'fabrication_year'",
            r'"fabrication_year"',
            r"'Description'",
            r'"Description"',
            r"'SKU'",
            r'"SKU"',
            
            # Old database columns
            r"\bvin_make\b",
            r"\bvin_year\b", 
            r"\bvin_series\b",
            r"\boriginal_description\b",
            r"\bnormalized_description\b",
            r"'sku'",
            r'"sku"',
            
            # Old Maestro columns
            r"\bVIN_Make\b",
            r"\bVIN_Year\b",
            r"\bVIN_Series_Trim\b",
            r"\bOriginal_Description_Input\b",
            r"\bNormalized_Description_Input\b",
            r"\bConfirmed_SKU\b",
            
            # Old variable patterns
            r"details\['Make'\]",
            r"details\['Model Year'\]",
            r"details\['fabrication_year'\]",
            r"details\['Description'\]",
            r"details\['SKU'\]",
            
            # Old DataFrame patterns
            r"df\['Make'\]",
            r"df\['Model Year'\]",
            r"df\['fabrication_year'\]",
            r"df\['Description'\]",
            r"df\['SKU'\]",
            r"df\['sku'\]",
        ]
        
        issues = []
        
        for pattern in old_patterns:
            matches = re.findall(pattern, content)
            if matches:
                issues.append(f"Found {len(matches)} occurrences of '{pattern}'")
        
        return issues
        
    except Exception as e:
        return [f"Error reading file: {e}"]

def check_correct_patterns(file_path):
    """Check that file contains the CORRECT field names"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # CORRECT field names that SHOULD exist
        correct_patterns = [
            r"'maker'",
            r"'series'", 
            r"'model'",
            r"'referencia'",
            r"'descripcion'"
        ]
        
        found_correct = []
        
        for pattern in correct_patterns:
            matches = re.findall(pattern, content)
            if matches:
                found_correct.append(f"✅ {pattern}: {len(matches)} occurrences")
        
        return found_correct
        
    except Exception as e:
        return [f"Error reading file: {e}"]

def main():
    """Main validation function"""
    print("="*80)
    print("FINAL FIELD NAME CONSISTENCY VALIDATION")
    print("="*80)
    print("Checking ALL files for consistency with original consolidado.json:")
    print("  ✅ SHOULD HAVE: maker, series, model, referencia, descripcion")
    print("  ❌ SHOULD NOT HAVE: Make, Year, fabrication_year, Description, SKU")
    print("="*80)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    # Find ALL files to check
    files_to_check = []
    
    # Python files
    python_patterns = [
        "src/*.py",
        "performance_improvements/**/*.py",
        "*.py"
    ]
    
    for pattern in python_patterns:
        files = glob.glob(str(project_root / pattern), recursive=True)
        files_to_check.extend(files)
    
    # Documentation files
    doc_patterns = [
        "docs/*.md",
        "*.md"
    ]
    
    for pattern in doc_patterns:
        files = glob.glob(str(project_root / pattern), recursive=True)
        files_to_check.extend(files)
    
    # Remove duplicates and filter
    files_to_check = sorted(list(set(files_to_check)))
    files_to_check = [f for f in files_to_check if 
                     "TEMPORARY SCRIPTS" not in f and 
                     "_backup" not in f and
                     "__pycache__" not in f]
    
    print(f"\nChecking {len(files_to_check)} files for consistency...")
    
    total_issues = 0
    files_with_issues = 0
    files_with_correct_patterns = 0
    
    for file_path in files_to_check:
        print(f"\n📁 {os.path.basename(file_path)}")
        
        # Check for old patterns (issues)
        issues = check_file_consistency(file_path)
        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            print(f"  ❌ ISSUES FOUND:")
            for issue in issues:
                print(f"     - {issue}")
        
        # Check for correct patterns
        correct = check_correct_patterns(file_path)
        if correct:
            files_with_correct_patterns += 1
            print(f"  ✅ CORRECT PATTERNS:")
            for pattern in correct:
                print(f"     - {pattern}")
        
        if not issues and not correct:
            print(f"  ⏭️  No field name patterns found")
    
    print("\n" + "="*80)
    print("FINAL VALIDATION SUMMARY")
    print("="*80)
    print(f"Total files checked: {len(files_to_check)}")
    print(f"Files with issues: {files_with_issues}")
    print(f"Total issues found: {total_issues}")
    print(f"Files with correct patterns: {files_with_correct_patterns}")
    
    if total_issues == 0:
        print("\n🎉 PERFECT CONSISTENCY ACHIEVED!")
        print("✅ ALL files use the correct original consolidado.json field names")
        print("✅ No old field names found anywhere in the project")
        print("\n🚀 The project is ready for use with 100% consistent naming!")
    else:
        print(f"\n⚠️  CONSISTENCY ISSUES FOUND")
        print(f"❌ {total_issues} issues need to be fixed in {files_with_issues} files")
        print("Please review the issues above and fix them manually")
    
    return total_issues == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
