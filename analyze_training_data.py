#!/usr/bin/env python3
"""
Analyze the training data to understand why training might be completing too quickly
"""

import sqlite3
import pandas as pd
import os

def analyze_training_data():
    """Analyze the data that will be used for training"""
    
    db_path = "data/fixacar_history.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    print("🔍 ANALYZING TRAINING DATA")
    print("=" * 60)
    
    conn = sqlite3.connect(db_path)
    
    # Total records
    total_query = "SELECT COUNT(*) as total FROM historical_parts WHERE sku IS NOT NULL"
    total_df = pd.read_sql_query(total_query, conn)
    total_records = total_df.iloc[0]['total']
    print(f"📊 Total records with SKU: {total_records:,}")
    
    # SKU distribution
    sku_query = "SELECT sku, COUNT(*) as frequency FROM historical_parts WHERE sku IS NOT NULL GROUP BY sku ORDER BY frequency DESC"
    sku_df = pd.read_sql_query(sku_query, conn)
    
    print(f"\n📋 SKU Distribution:")
    print(f"  Total unique SKUs: {len(sku_df):,}")
    print(f"  Most common SKU: {sku_df.iloc[0]['sku']} ({sku_df.iloc[0]['frequency']:,} times)")
    print(f"  Least common SKU: {sku_df.iloc[-1]['sku']} ({sku_df.iloc[-1]['frequency']:,} times)")
    
    # Frequency analysis
    freq_analysis = sku_df['frequency'].describe()
    print(f"\n📈 Frequency Statistics:")
    print(f"  Mean frequency: {freq_analysis['mean']:.1f}")
    print(f"  Median frequency: {freq_analysis['50%']:.1f}")
    print(f"  75th percentile: {freq_analysis['75%']:.1f}")
    print(f"  95th percentile: {sku_df['frequency'].quantile(0.95):.1f}")
    
    # Check different MIN_SKU_FREQUENCY thresholds
    print(f"\n🔧 Impact of MIN_SKU_FREQUENCY:")
    for min_freq in [1, 2, 3, 5, 10]:
        filtered_skus = sku_df[sku_df['frequency'] >= min_freq]
        total_records_kept = filtered_skus['frequency'].sum()
        print(f"  Min freq {min_freq}: {len(filtered_skus):,} SKUs, {total_records_kept:,} records ({total_records_kept/total_records*100:.1f}%)")
    
    # Top 20 most common SKUs
    print(f"\n🏆 Top 20 Most Common SKUs:")
    for i, row in sku_df.head(20).iterrows():
        print(f"  {i+1:2d}. {row['sku']}: {row['frequency']:,} times")
    
    # Check for potential overfitting indicators
    very_common_skus = sku_df[sku_df['frequency'] > 1000]
    if len(very_common_skus) > 0:
        print(f"\n⚠️  Very Common SKUs (>1000 occurrences): {len(very_common_skus)}")
        print("   These might cause the model to converge quickly:")
        for _, row in very_common_skus.head(10).iterrows():
            print(f"     {row['sku']}: {row['frequency']:,} times")
    
    # Check description diversity
    desc_query = "SELECT COUNT(DISTINCT normalized_description) as unique_descriptions FROM historical_parts WHERE sku IS NOT NULL"
    desc_df = pd.read_sql_query(desc_query, conn)
    unique_descriptions = desc_df.iloc[0]['unique_descriptions']
    
    print(f"\n📝 Description Analysis:")
    print(f"  Unique descriptions: {unique_descriptions:,}")
    print(f"  Records per description: {total_records/unique_descriptions:.1f}")
    
    # Sample some descriptions
    sample_query = "SELECT DISTINCT normalized_description FROM historical_parts WHERE sku IS NOT NULL LIMIT 10"
    sample_df = pd.read_sql_query(sample_query, conn)
    print(f"\n📄 Sample Descriptions:")
    for i, row in sample_df.iterrows():
        desc = row['normalized_description']
        print(f"  {i+1:2d}. {desc[:60]}{'...' if len(desc) > 60 else ''}")
    
    conn.close()
    
    # Recommendations
    print(f"\n💡 TRAINING RECOMMENDATIONS:")
    print("=" * 60)
    
    if len(very_common_skus) > 10:
        print("⚠️  HIGH RISK: Many very common SKUs detected")
        print("   - Model may converge too quickly")
        print("   - Consider increasing patience and epochs")
        print("   - Use smaller learning rate")
    
    if unique_descriptions < total_records * 0.1:
        print("⚠️  LOW DIVERSITY: Few unique descriptions relative to records")
        print("   - Model may overfit quickly")
        print("   - Consider data augmentation or regularization")
    
    print("✅ CURRENT OPTIMIZED SETTINGS:")
    print("   - MIN_SKU_FREQUENCY: 2 (includes more SKUs)")
    print("   - EPOCHS: 100 (increased from 50)")
    print("   - PATIENCE: 20 (increased from 10)")
    print("   - BATCH_SIZE: 64 (reduced from 128)")
    print("   - LEARNING_RATE: 0.0005 (reduced from 0.001)")
    print("   - HIDDEN_DIM: 128 (restored from 64)")

if __name__ == "__main__":
    analyze_training_data()
