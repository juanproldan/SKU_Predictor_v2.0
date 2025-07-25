#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Optimization Recommendations for SKU Predictor
Detailed implementation examples for caching and optimization strategies
"""

import hashlib
import json
import time
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import sqlite3
import pickle

# ============================================================================
# 1. CACHING FOR FREQUENT SKU PREDICTIONS
# ============================================================================

class SKUPredictionCache:
    """
    Multi-level caching system for SKU predictions
    
    Level 1: In-memory cache for current session (fastest)
    Level 2: SQLite cache for persistent storage across sessions
    Level 3: File-based cache for backup
    """
    
    def __init__(self, max_memory_cache=1000, cache_db_path="cache/prediction_cache.db"):
        self.memory_cache = {}  # In-memory cache
        self.max_memory_cache = max_memory_cache
        self.cache_db_path = cache_db_path
        self.hit_count = 0
        self.miss_count = 0
        
        # Initialize persistent cache database
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_cache (
            cache_key TEXT PRIMARY KEY,
            vin_make TEXT,
            vin_year TEXT,
            vin_series TEXT,
            description TEXT,
            predictions TEXT,  -- JSON string of predictions
            confidence_scores TEXT,  -- JSON string of confidence scores
            timestamp REAL,
            hit_count INTEGER DEFAULT 1
        )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON prediction_cache(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_cache_key(self, vin_make: str, vin_year: str, vin_series: str, description: str) -> str:
        """Generate unique cache key for prediction inputs"""
        # Normalize inputs for consistent caching
        normalized_input = f"{vin_make.lower()}|{vin_year}|{vin_series.lower()}|{description.lower().strip()}"
        return hashlib.md5(normalized_input.encode()).hexdigest()
    
    def get_cached_prediction(self, vin_make: str, vin_year: str, vin_series: str, description: str) -> Optional[Dict]:
        """
        Retrieve cached prediction if available
        Returns: Dict with predictions and confidence scores, or None if not cached
        """
        cache_key = self._generate_cache_key(vin_make, vin_year, vin_series, description)
        
        # Level 1: Check in-memory cache first (fastest)
        if cache_key in self.memory_cache:
            self.hit_count += 1
            print(f"  🚀 Cache HIT (Memory): {description[:30]}...")
            return self.memory_cache[cache_key]
        
        # Level 2: Check persistent cache database
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT predictions, confidence_scores, hit_count 
        FROM prediction_cache 
        WHERE cache_key = ? AND timestamp > ?
        ''', (cache_key, time.time() - 86400))  # Cache valid for 24 hours
        
        result = cursor.fetchone()
        
        if result:
            predictions_json, confidence_json, hit_count = result
            cached_data = {
                'predictions': json.loads(predictions_json),
                'confidence_scores': json.loads(confidence_json)
            }
            
            # Update hit count and add to memory cache
            cursor.execute('''
            UPDATE prediction_cache 
            SET hit_count = hit_count + 1 
            WHERE cache_key = ?
            ''', (cache_key,))
            
            conn.commit()
            conn.close()
            
            # Add to memory cache for faster future access
            self._add_to_memory_cache(cache_key, cached_data)
            
            self.hit_count += 1
            print(f"  🚀 Cache HIT (Database): {description[:30]}... (used {hit_count + 1} times)")
            return cached_data
        
        conn.close()
        self.miss_count += 1
        print(f"  ❌ Cache MISS: {description[:30]}...")
        return None
    
    def cache_prediction(self, vin_make: str, vin_year: str, vin_series: str, description: str, 
                        predictions: List[Dict], confidence_scores: List[float]):
        """Cache a new prediction result"""
        cache_key = self._generate_cache_key(vin_make, vin_year, vin_series, description)
        
        cached_data = {
            'predictions': predictions,
            'confidence_scores': confidence_scores
        }
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, cached_data)
        
        # Add to persistent cache
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO prediction_cache 
        (cache_key, vin_make, vin_year, vin_series, description, predictions, confidence_scores, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cache_key, vin_make, vin_year, vin_series, description,
            json.dumps(predictions), json.dumps(confidence_scores), time.time()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"  💾 Cached prediction: {description[:30]}...")
    
    def _add_to_memory_cache(self, cache_key: str, data: Dict):
        """Add item to memory cache with LRU eviction"""
        if len(self.memory_cache) >= self.max_memory_cache:
            # Remove oldest item (simple FIFO, could implement proper LRU)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = data
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_cache_size': len(self.memory_cache)
        }
    
    def clear_old_cache(self, days_old: int = 7):
        """Clear cache entries older than specified days"""
        cutoff_time = time.time() - (days_old * 86400)
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM prediction_cache WHERE timestamp < ?', (cutoff_time,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"🧹 Cleared {deleted_count} old cache entries (older than {days_old} days)")


# ============================================================================
# 2. DATABASE QUERY OPTIMIZATION
# ============================================================================

class DatabaseOptimizer:
    """
    Database optimization strategies for better query performance
    """
    
    @staticmethod
    def create_optimized_indexes(db_path: str):
        """Create optimized indexes for common query patterns"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Indexes for SKU prediction queries
        optimization_queries = [
            # Composite index for exact matching (most common query)
            '''CREATE INDEX IF NOT EXISTS idx_exact_match 
               ON processed_consolidado(vin_make, vin_year, vin_series, normalized_description)''',
            
            # Index for VIN-based queries
            '''CREATE INDEX IF NOT EXISTS idx_vin_lookup 
               ON processed_consolidado(vin_number)''',
            
            # Index for SKU frequency analysis
            '''CREATE INDEX IF NOT EXISTS idx_sku_frequency 
               ON processed_consolidado(sku, vin_make, vin_year)''',
            
            # Index for description-based fuzzy matching
            '''CREATE INDEX IF NOT EXISTS idx_description_search 
               ON processed_consolidado(normalized_description, sku)''',
            
            # Index for date-based queries (for incremental training)
            '''CREATE INDEX IF NOT EXISTS idx_date_queries 
               ON processed_consolidado(vin_year, vin_make)'''
        ]
        
        for query in optimization_queries:
            try:
                cursor.execute(query)
                print(f"✅ Created index: {query.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                print(f"❌ Failed to create index: {e}")
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def analyze_query_performance(db_path: str):
        """Analyze database query performance and suggest optimizations"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable query planning
        cursor.execute("PRAGMA query_planner = ON")
        
        # Test common query patterns
        test_queries = [
            "SELECT COUNT(*) FROM processed_consolidado",
            "SELECT * FROM processed_consolidado WHERE vin_make = 'TOYOTA' LIMIT 10",
            "SELECT sku, COUNT(*) FROM processed_consolidado GROUP BY sku ORDER BY COUNT(*) DESC LIMIT 10"
        ]
        
        for query in test_queries:
            start_time = time.time()
            cursor.execute(query)
            results = cursor.fetchall()
            end_time = time.time()
            
            print(f"Query: {query[:50]}...")
            print(f"  Time: {(end_time - start_time)*1000:.2f}ms")
            print(f"  Results: {len(results)} rows")
            print()
        
        conn.close()


# ============================================================================
# 3. PREDICTION RESULT CACHING FOR IDENTICAL INPUTS
# ============================================================================

class PredictionResultCache:
    """
    Specialized cache for complete prediction results
    Handles the full prediction pipeline output
    """
    
    def __init__(self):
        self.session_cache = {}  # Cache for current session
        self.cache_file = "cache/prediction_results.pkl"
        self.persistent_cache = self._load_persistent_cache()
    
    def _load_persistent_cache(self) -> Dict:
        """Load persistent cache from file"""
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            return {}
    
    def _save_persistent_cache(self):
        """Save cache to file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.persistent_cache, f)
        except Exception as e:
            print(f"Warning: Could not save prediction cache: {e}")
    
    def get_cached_result(self, input_hash: str) -> Optional[Dict]:
        """Get cached prediction result"""
        # Check session cache first
        if input_hash in self.session_cache:
            return self.session_cache[input_hash]
        
        # Check persistent cache
        if input_hash in self.persistent_cache:
            # Move to session cache for faster access
            result = self.persistent_cache[input_hash]
            self.session_cache[input_hash] = result
            return result
        
        return None
    
    def cache_result(self, input_hash: str, result: Dict):
        """Cache a prediction result"""
        self.session_cache[input_hash] = result
        self.persistent_cache[input_hash] = result
        
        # Periodically save to file (every 10 new entries)
        if len(self.session_cache) % 10 == 0:
            self._save_persistent_cache()


# ============================================================================
# USAGE EXAMPLE IN MAIN APPLICATION
# ============================================================================

def optimized_sku_prediction_example():
    """
    Example of how to integrate caching into the main SKU prediction function
    """
    
    # Initialize cache
    cache = SKUPredictionCache()
    
    def predict_sku_with_cache(vin_make: str, vin_year: str, vin_series: str, description: str):
        """Enhanced SKU prediction with caching"""
        
        # Check cache first
        cached_result = cache.get_cached_prediction(vin_make, vin_year, vin_series, description)
        if cached_result:
            return cached_result
        
        # If not cached, run normal prediction pipeline
        print(f"  🔄 Running full prediction pipeline for: {description[:30]}...")
        
        # Simulate prediction pipeline (replace with actual prediction code)
        predictions = [
            {"sku": "ABC123", "source": "Maestro", "confidence": 0.95},
            {"sku": "DEF456", "source": "Neural Network", "confidence": 0.82}
        ]
        confidence_scores = [0.95, 0.82]
        
        # Cache the result
        cache.cache_prediction(vin_make, vin_year, vin_series, description, predictions, confidence_scores)
        
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores
        }
    
    # Example usage
    result1 = predict_sku_with_cache("TOYOTA", "2020", "COROLLA", "parachoques delantero")
    result2 = predict_sku_with_cache("TOYOTA", "2020", "COROLLA", "parachoques delantero")  # Should be cached
    
    # Print cache statistics
    stats = cache.get_cache_stats()
    print(f"\n📊 Cache Performance: {stats['hit_rate_percent']:.1f}% hit rate")
    print(f"   Hits: {stats['hit_count']}, Misses: {stats['miss_count']}")


if __name__ == "__main__":
    print("🚀 Performance Optimization Examples")
    print("=" * 50)
    
    # Example 1: Database optimization
    print("\n1. Database Optimization:")
    # DatabaseOptimizer.create_optimized_indexes("Source_Files/processed_consolidado.db")
    
    # Example 2: Caching demonstration
    print("\n2. Caching Demonstration:")
    optimized_sku_prediction_example()
