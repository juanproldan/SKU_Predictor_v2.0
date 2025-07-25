#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consolidado JSON Comparison Script

Compares two Consolidado.json files to analyze differences in:
- Total number of records
- Record structure changes
- Data growth analysis
- New vs existing records
- Statistical summary

Author: Augment Agent
Date: 2025-07-25
"""

import json
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter

def load_json_file(file_path):
    """Load and parse JSON file."""
    print(f"Loading: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} records")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to load file: {e}")
        return None

def analyze_record_structure(records, file_name):
    """Analyze the structure of records in the dataset."""
    print(f"\n📊 Analyzing structure of {file_name}:")

    if not records:
        print("  No records to analyze")
        return {}

    # Sample first record to understand structure
    sample_record = records[0]
    print(f"  📝 Sample record keys: {list(sample_record.keys())}")

    # Count total items across all records - check both 'Items' and 'items'
    total_items = 0
    records_with_items = 0

    for record in records:
        items_list = None
        if 'Items' in record and isinstance(record['Items'], list):
            items_list = record['Items']
        elif 'items' in record and isinstance(record['items'], list):
            items_list = record['items']

        if items_list is not None:
            item_count = len(items_list)
            total_items += item_count
            if item_count > 0:
                records_with_items += 1

    print(f"  📦 Total records: {len(records):,}")
    print(f"  📦 Records with items: {records_with_items:,}")
    print(f"  📦 Total items: {total_items:,}")
    print(f"  📦 Average items per record: {total_items/len(records):.2f}")

    return {
        'total_records': len(records),
        'total_items': total_items,
        'records_with_items': records_with_items,
        'avg_items_per_record': total_items/len(records) if records else 0
    }

def extract_record_identifiers(records):
    """Extract unique identifiers from records for comparison."""
    identifiers = set()

    for record in records:
        # Try different possible identifier fields
        record_id = None

        if '_id' in record:
            record_id = record['_id']
        elif 'ID' in record:
            record_id = record['ID']
        elif 'id' in record:
            record_id = record['id']
        elif 'quote' in record:
            record_id = record['quote']
        elif 'BidID' in record:
            record_id = record['BidID']
        elif 'bid_id' in record:
            record_id = record['bid_id']

        if record_id:
            identifiers.add(str(record_id))

    return identifiers

def analyze_date_ranges(records, file_name):
    """Analyze date ranges in the dataset."""
    print(f"\n📅 Date analysis for {file_name}:")

    dates = []
    date_formats = set()

    for record in records:
        if 'date' in record and record['date']:
            date_value = record['date']
            date_formats.add(str(type(date_value)))

            # Handle different date formats
            if isinstance(date_value, dict):
                # MongoDB date format like {"$date": "2023-01-01T00:00:00.000Z"}
                if '$date' in date_value:
                    dates.append(date_value['$date'])
                else:
                    dates.append(str(date_value))
            else:
                dates.append(str(date_value))

    print(f"  📅 Date formats found: {date_formats}")

    if dates:
        dates.sort()
        print(f"  📅 Earliest date: {dates[0]}")
        print(f"  📅 Latest date: {dates[-1]}")
        print(f"  📅 Total records with dates: {len(dates):,}")
    else:
        print("  ⚠️ No date information found")

    return dates

def compare_datasets(old_data, new_data):
    """Compare two datasets and provide detailed analysis."""
    print("\n" + "="*60)
    print("🔍 DETAILED COMPARISON ANALYSIS")
    print("="*60)
    
    # Basic statistics
    old_stats = analyze_record_structure(old_data, "Old Consolidado")
    new_stats = analyze_record_structure(new_data, "New Consolidado")
    
    # Calculate differences
    record_diff = new_stats['total_records'] - old_stats['total_records']
    item_diff = new_stats['total_items'] - old_stats['total_items']

    print(f"\n📈 GROWTH ANALYSIS:")
    print(f"  📦 Record growth: {record_diff:,} ({record_diff/old_stats['total_records']*100:.2f}% increase)")

    if old_stats['total_items'] > 0:
        print(f"  📦 Item growth: {item_diff:,} ({item_diff/old_stats['total_items']*100:.2f}% increase)")
    else:
        print(f"  📦 Item growth: {item_diff:,} (Cannot calculate percentage - old file has 0 items)")
    
    # Try to identify new vs existing records
    print(f"\n🔍 RECORD OVERLAP ANALYSIS:")
    old_ids = extract_record_identifiers(old_data)
    new_ids = extract_record_identifiers(new_data)
    
    if old_ids and new_ids:
        common_ids = old_ids.intersection(new_ids)
        new_only_ids = new_ids - old_ids
        old_only_ids = old_ids - new_ids
        
        print(f"  🔄 Common records: {len(common_ids):,}")
        print(f"  ➕ New records: {len(new_only_ids):,}")
        print(f"  ➖ Removed records: {len(old_only_ids):,}")
        
        if len(new_only_ids) > 0:
            print(f"  📊 New record percentage: {len(new_only_ids)/len(new_ids)*100:.2f}%")
    else:
        print("  ⚠️ Could not identify record IDs for overlap analysis")
    
    # Sample comparison
    print(f"\n📋 SAMPLE RECORD COMPARISON:")
    if old_data and new_data:
        print("  Old file sample record:")
        print(f"    Keys: {list(old_data[0].keys())}")
        print("  New file sample record:")
        print(f"    Keys: {list(new_data[0].keys())}")
        
        # Check if structure is the same
        old_keys = set(old_data[0].keys())
        new_keys = set(new_data[0].keys())
        
        if old_keys == new_keys:
            print("  ✅ Record structure is identical")
        else:
            added_keys = new_keys - old_keys
            removed_keys = old_keys - new_keys
            if added_keys:
                print(f"  ➕ New keys: {list(added_keys)}")
            if removed_keys:
                print(f"  ➖ Removed keys: {list(removed_keys)}")

def main():
    """Main comparison function."""
    print("🔍 CONSOLIDADO JSON COMPARISON TOOL")
    print("="*50)
    
    # File paths
    old_file = r"C:\Users\juanp\Downloads\oldConsolidado.json"
    new_file = r"C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0\Source_Files\Consolidado.json"
    
    print(f"📁 Old file: {old_file}")
    print(f"📁 New file: {new_file}")
    
    # Load both files
    old_data = load_json_file(old_file)
    new_data = load_json_file(new_file)
    
    if old_data is None or new_data is None:
        print("❌ Cannot proceed with comparison - one or both files failed to load")
        return
    
    # Analyze date ranges
    old_dates = analyze_date_ranges(old_data, "Old Consolidado")
    new_dates = analyze_date_ranges(new_data, "New Consolidado")

    # Perform comparison
    compare_datasets(old_data, new_data)

    print(f"\n✅ Comparison completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
