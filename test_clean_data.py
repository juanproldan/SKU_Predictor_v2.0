#!/usr/bin/env python3
"""
Test the cleaned data query to see how much valid data we have
"""

import sqlite3
import pandas as pd

def test_clean_data():
    """Test the impact of cleaning empty SKUs"""
    
    conn = sqlite3.connect('data/fixacar_history.db')
    
    # Original query (includes empty SKUs)
    original_query = "SELECT COUNT(*) as total FROM historical_parts WHERE sku IS NOT NULL"
    original_df = pd.read_sql_query(original_query, conn)
    original_count = original_df.iloc[0]['total']
    
    # Cleaned query (excludes empty SKUs)
    clean_query = "SELECT COUNT(*) as total FROM historical_parts WHERE sku IS NOT NULL AND TRIM(sku) != '' AND LENGTH(TRIM(sku)) > 0"
    clean_df = pd.read_sql_query(clean_query, conn)
    clean_count = clean_df.iloc[0]['total']
    
    # Check what we're removing
    empty_count = original_count - clean_count
    
    print("🧹 DATA CLEANING ANALYSIS")
    print("=" * 50)
    print(f"📊 Original records (sku IS NOT NULL): {original_count:,}")
    print(f"✅ Clean records (non-empty SKUs): {clean_count:,}")
    print(f"🗑️  Empty/blank SKUs removed: {empty_count:,}")
    print(f"📈 Data retention: {clean_count/original_count*100:.1f}%")
    
    # Check the top SKUs after cleaning
    top_skus_query = """
    SELECT sku, COUNT(*) as frequency 
    FROM historical_parts 
    WHERE sku IS NOT NULL AND TRIM(sku) != '' AND LENGTH(TRIM(sku)) > 0 
    GROUP BY sku 
    ORDER BY frequency DESC 
    LIMIT 10
    """
    top_skus_df = pd.read_sql_query(top_skus_query, conn)
    
    print(f"\n🏆 Top 10 SKUs after cleaning:")
    for i, row in top_skus_df.iterrows():
        print(f"  {i+1:2d}. {row['sku']}: {row['frequency']:,} times")
    
    # Check unique SKUs after cleaning
    unique_skus_query = """
    SELECT COUNT(DISTINCT sku) as unique_skus 
    FROM historical_parts 
    WHERE sku IS NOT NULL AND TRIM(sku) != '' AND LENGTH(TRIM(sku)) > 0
    """
    unique_df = pd.read_sql_query(unique_skus_query, conn)
    unique_skus = unique_df.iloc[0]['unique_skus']
    
    print(f"\n📋 After cleaning:")
    print(f"  Unique SKUs: {unique_skus:,}")
    print(f"  Average frequency: {clean_count/unique_skus:.1f}")
    
    conn.close()
    
    if empty_count > 0:
        print(f"\n✅ PROBLEM IDENTIFIED AND FIXED!")
        print(f"   The {empty_count:,} empty SKUs were causing quick convergence")
        print(f"   Training should now take much longer with {clean_count:,} clean records")
    else:
        print(f"\n❓ No empty SKUs found - the issue might be elsewhere")

if __name__ == "__main__":
    test_clean_data()
