#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Series Lookup Tool

This tool helps users find the correct series names from the database
based on maker, model year, and partial series input. It provides
fuzzy matching and suggestions to improve SKU prediction accuracy.
"""

import os
import sys
import sqlite3
from difflib import SequenceMatcher
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def similarity(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

class SeriesLookupTool:
    """Tool for finding correct series names in the database."""
    
    def __init__(self):
        from unified_consolidado_processor import get_base_path
        base_path = get_base_path()
        self.db_path = os.path.join(base_path, "Source_Files", "processed_consolidado.db")
        
    def find_series(self, maker, model_year=None, partial_series=None, limit=10):
        """
        Find series names matching the criteria.
        
        Args:
            maker: Vehicle manufacturer (e.g., "Hyundai", "Toyota")
            model_year: Vehicle year (e.g., "2018", 2018)
            partial_series: Partial series name (e.g., "Tucson", "Unknown")
            limit: Maximum number of results
            
        Returns:
            List of tuples: (series_name, record_count, similarity_score)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query based on provided parameters
        query_parts = ["SELECT series, COUNT(*) as count FROM processed_consolidado WHERE 1=1"]
        params = []
        
        # Add maker filter
        if maker:
            query_parts.append("AND LOWER(maker) LIKE LOWER(?)")
            params.append(f"%{maker}%")
        
        # Add model year filter
        if model_year:
            query_parts.append("AND model = ?")
            params.append(str(model_year))
        
        # Add partial series filter if provided and not "Unknown"
        if partial_series and partial_series.lower() != "unknown":
            query_parts.append("AND LOWER(series) LIKE LOWER(?)")
            params.append(f"%{partial_series}%")
        
        query_parts.append("GROUP BY series ORDER BY count DESC")
        if limit:
            query_parts.append(f"LIMIT {limit * 3}")  # Get more for similarity filtering
        
        query = " ".join(query_parts)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Calculate similarity scores if partial_series provided
        scored_results = []
        for series, count in results:
            if partial_series and partial_series.lower() != "unknown":
                sim_score = similarity(partial_series, series)
            else:
                sim_score = 1.0  # No partial series, all equally valid
            
            scored_results.append((series, count, sim_score))
        
        # Sort by similarity score (desc) then by count (desc)
        scored_results.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        conn.close()
        return scored_results[:limit]
    
    def get_maker_stats(self):
        """Get statistics about makers in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT maker, COUNT(*) as count, COUNT(DISTINCT series) as series_count
            FROM processed_consolidado 
            GROUP BY maker 
            ORDER BY count DESC 
            LIMIT 20
        """)
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_year_range_for_series(self, maker, series):
        """Get the year range for a specific maker/series combination."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MIN(CAST(model AS INTEGER)) as min_year, 
                   MAX(CAST(model AS INTEGER)) as max_year,
                   COUNT(*) as total_records
            FROM processed_consolidado 
            WHERE LOWER(maker) LIKE LOWER(?) 
            AND series = ?
            AND model REGEXP '^[0-9]+$'
        """, (f"%{maker}%", series))
        
        result = cursor.fetchone()
        conn.close()
        return result
    
    def suggest_alternatives(self, maker, model_year, failed_series):
        """Suggest alternative series when the provided one doesn't work."""
        print(f"\n🔍 Searching alternatives for: {maker} {model_year} '{failed_series}'")
        
        # Try exact maker match first
        exact_matches = self.find_series(maker, model_year, limit=5)
        if exact_matches:
            print(f"✅ Found {len(exact_matches)} series for {maker} {model_year}:")
            for series, count, sim in exact_matches:
                print(f"  - '{series}': {count:,} records (similarity: {sim:.2f})")
            return exact_matches
        
        # Try without year
        year_flexible = self.find_series(maker, limit=10)
        if year_flexible:
            print(f"📅 No {model_year} data, but found series for {maker}:")
            for series, count, sim in year_flexible[:5]:
                year_info = self.get_year_range_for_series(maker, series)
                if year_info and year_info[0] and year_info[1]:
                    year_range = f"{year_info[0]}-{year_info[1]}"
                else:
                    year_range = "unknown years"
                print(f"  - '{series}': {count:,} records ({year_range})")
            return year_flexible[:5]
        
        # Try fuzzy maker matching
        print(f"🔄 Trying fuzzy maker matching...")
        fuzzy_makers = self.find_similar_makers(maker)
        if fuzzy_makers:
            print(f"💡 Did you mean one of these makers?")
            for similar_maker, count in fuzzy_makers:
                print(f"  - {similar_maker}: {count:,} records")
        
        return []
    
    def find_similar_makers(self, target_maker):
        """Find makers similar to the target maker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT maker, COUNT(*) as count FROM processed_consolidado GROUP BY maker")
        all_makers = cursor.fetchall()
        
        similar_makers = []
        for maker, count in all_makers:
            if maker and similarity(target_maker, maker) > 0.6:
                similar_makers.append((maker, count))
        
        similar_makers.sort(key=lambda x: x[1], reverse=True)
        conn.close()
        return similar_makers[:5]

def interactive_series_lookup():
    """Interactive series lookup tool."""
    tool = SeriesLookupTool()
    
    print("🔍 Series Lookup Tool")
    print("=" * 50)
    print("This tool helps you find the correct series names for SKU prediction.")
    print("Type 'quit' to exit, 'stats' for database statistics.\n")
    
    while True:
        try:
            print("\n" + "─" * 50)
            maker = input("Enter maker (e.g., Hyundai, Toyota): ").strip()
            
            if maker.lower() == 'quit':
                break
            elif maker.lower() == 'stats':
                print("\n📊 Database Statistics:")
                stats = tool.get_maker_stats()
                for maker_name, count, series_count in stats:
                    print(f"  {maker_name}: {count:,} records, {series_count} series")
                continue
            
            if not maker:
                print("❌ Please enter a maker name.")
                continue
            
            model_year = input("Enter model year (optional, e.g., 2018): ").strip()
            if model_year and not model_year.isdigit():
                model_year = None
            
            partial_series = input("Enter series hint (optional, e.g., Tucson): ").strip()
            if not partial_series:
                partial_series = None
            
            print(f"\n🔍 Searching for: {maker} {model_year or 'any year'} {partial_series or 'any series'}")
            
            results = tool.find_series(maker, model_year, partial_series)
            
            if results:
                print(f"\n✅ Found {len(results)} matching series:")
                for i, (series, count, sim) in enumerate(results, 1):
                    sim_indicator = f"(sim: {sim:.2f})" if partial_series else ""
                    print(f"  {i}. '{series}': {count:,} records {sim_indicator}")
                    
                    # Show year range for this series
                    year_info = tool.get_year_range_for_series(maker, series)
                    if year_info and year_info[0] and year_info[1]:
                        print(f"     Years: {year_info[0]}-{year_info[1]}")
            else:
                print("❌ No matching series found.")
                tool.suggest_alternatives(maker, model_year, partial_series or "unknown")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def test_user_examples():
    """Test the tool with user's specific examples."""
    tool = SeriesLookupTool()
    
    print("🧪 Testing User Examples")
    print("=" * 40)
    
    test_cases = [
        ("Hyundai", "2018", "Unknown"),
        ("Hyundai", "2018", "Tucson"),
        ("MAZDA", "2016", "BT50"),
        ("Toyota", "2016", "Tucson"),  # Should suggest Hyundai
    ]
    
    for maker, year, series in test_cases:
        print(f"\n--- Testing: {maker} {year} {series} ---")
        results = tool.find_series(maker, year, series if series != "Unknown" else None)
        
        if results:
            print(f"✅ Found {len(results)} matches:")
            for series_name, count, sim in results[:3]:
                print(f"  - '{series_name}': {count:,} records")
        else:
            tool.suggest_alternatives(maker, year, series)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_user_examples()
    else:
        interactive_series_lookup()
