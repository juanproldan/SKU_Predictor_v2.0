#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick Performance Test - Focus on database and cache performance
Simple validation without complex data loading

Author: Augment Agent
Date: 2025-07-25
"""

import os
import sys
import time
import sqlite3
import json
import numpy as np

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, 'src')

# Add all necessary paths
paths_to_add = [project_root, src_path]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"📁 Project root: {project_root}")
print(f"📁 Source path: {src_path}")
print(f"📁 Current dir: {current_dir}")

def test_database_performance():
    """Test database performance with real queries"""
    print("🗄️ Testing Database Performance...")
    
    db_path = os.path.join(project_root, "Source_Files", "processed_consolidado.db")
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    conn = sqlite3.connect(db_path)
    
    # Test queries
    test_queries = [
        ("Count All Records", "SELECT COUNT(*) FROM processed_consolidado"),
        ("Sample Records", "SELECT * FROM processed_consolidado LIMIT 5"),
        ("Make Filter", "SELECT COUNT(*) FROM processed_consolidado WHERE maker = 'TOYOTA'"),
        ("Year Filter", "SELECT COUNT(*) FROM processed_consolidado WHERE model = '2020'"),
        ("SKU Count", "SELECT COUNT(DISTINCT referencia) FROM processed_consolidado WHERE referencia IS NOT NULL"),
        ("Description Search", "SELECT COUNT(*) FROM processed_consolidado WHERE normalized_descripcion LIKE '%parachoques%'")
    ]
    
    results = {}
    
    for name, query in test_queries:
        print(f"   🔍 {name}...")
        
        times = []
        for _ in range(3):  # Run 3 times for average
            start_time = time.time()
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        result_count = len(result)
        
        results[name] = {
            'avg_time_ms': avg_time,
            'result_count': result_count,
            'is_fast': avg_time < 100  # Under 100ms is good
        }
        
        status = "🚀" if avg_time < 100 else "⚠️"
        print(f"      {status} {avg_time:.2f}ms ({result_count} results)")
    
    conn.close()
    
    # Summary
    fast_queries = sum(1 for r in results.values() if r['is_fast'])
    total_queries = len(results)
    
    print(f"\n   📊 Database Performance Summary:")
    print(f"      Fast Queries: {fast_queries}/{total_queries}")
    print(f"      Performance Score: {fast_queries/total_queries*100:.1f}%")
    
    return results

def test_cache_system():
    """Test cache system functionality"""
    print("\n💾 Testing Cache System...")
    
    try:
        from performance_improvements.cache.referencia_prediction_cache import initialize_cache, get_cache
        
        # Initialize cache
        cache = initialize_cache()
        print("   ✅ Cache system initialized")
        
        # Test cache operations
        test_data = [
            ("TOYOTA_2020_COROLLA_parachoques delantero", {"referencia": "TEST001", "confidence": 0.9}),
            ("HONDA_2019_CIVIC_faro izquierdo", {"referencia": "TEST002", "confidence": 0.8}),
            ("NISSAN_2021_SENTRA_espejo derecho", {"referencia": "TEST003", "confidence": 0.7})
        ]
        
        # Test cache operations with proper method names
        print("   🔄 Testing cache operations...")
        cache_times = []

        # Test caching predictions
        for i, (key_parts, value) in enumerate(test_data):
            # Parse key parts (make_year_series_description)
            parts = key_parts.split('_')
            make, year, series = parts[0], parts[1], parts[2]
            description = '_'.join(parts[3:])

            start_time = time.time()
            cache.cache_prediction(make, year, series, description, [value], [value['confidence']], ['test'])
            end_time = time.time()
            cache_times.append((end_time - start_time) * 1000)

        avg_cache_time = np.mean(cache_times)
        print(f"      Average cache time: {avg_cache_time:.3f}ms")

        # Test cache retrieval operations
        print("   🔄 Testing cache retrieval...")
        get_times = []
        hits = 0

        for key_parts, expected_value in test_data:
            # Parse key parts
            parts = key_parts.split('_')
            make, year, series = parts[0], parts[1], parts[2]
            description = '_'.join(parts[3:])

            start_time = time.time()
            cached_value = cache.get_cached_prediction(make, year, series, description)
            end_time = time.time()
            get_times.append((end_time - start_time) * 1000)

            if cached_value is not None:
                hits += 1
        
        avg_get_time = np.mean(get_times)
        hit_rate = hits / len(test_data) * 100

        print(f"      Average retrieval time: {avg_get_time:.3f}ms")
        print(f"      Hit rate: {hit_rate:.1f}%")

        # Test cache statistics
        if hasattr(cache, 'get_cache_stats'):
            stats = cache.get_cache_stats()
            print(f"      Cache stats: {stats}")

        return {
            'available': True,
            'avg_cache_time_ms': avg_cache_time,
            'avg_get_time_ms': avg_get_time,
            'hit_rate_percent': hit_rate,
            'test_operations': len(test_data)
        }
        
    except Exception as e:
        print(f"   ❌ Cache system error: {e}")
        return {'available': False, 'error': str(e)}

def test_enhanced_text_processing():
    """Test enhanced text processing"""
    print("\n📝 Testing Enhanced Text Processing...")
    
    try:
        # Import standard text processing
        from utils.text_utils import normalize_text
        print("   ✅ Standard text processing imported")
        
        # Test descriptions
        test_descriptions = [
            "PARACHOQUES DEL TOYOTA",
            "FARO IZQ HONDA",
            "ESPEJO DER NISSAN",
            "VIDRIO PUER.DL.D.",
            "GUARDAFANGO TRAS DERECHO"
        ]
        
        print(f"   🔄 Testing with {len(test_descriptions)} descriptions...")
        
        # Test standard processing
        standard_times = []
        standard_results = []
        
        for desc in test_descriptions:
            start_time = time.time()
            result = normalize_text(desc)
            end_time = time.time()
            
            standard_times.append((end_time - start_time) * 1000)
            standard_results.append(result)
        
        avg_standard_time = np.mean(standard_times)
        print(f"      Standard processing: {avg_standard_time:.3f}ms avg")
        
        # Show some results
        print("      Sample results:")
        for i, (orig, processed) in enumerate(zip(test_descriptions[:3], standard_results[:3])):
            print(f"        '{orig}' → '{processed}'")
        
        # Try enhanced processing
        try:
            from performance_improvements.enhanced_text_processing.smart_text_processor import get_smart_processor
            smart_processor = get_smart_processor()
            
            enhanced_times = []
            enhanced_results = []
            
            for desc in test_descriptions:
                start_time = time.time()
                result = smart_processor.process_text_enhanced(desc)
                end_time = time.time()
                
                enhanced_times.append((end_time - start_time) * 1000)
                enhanced_results.append(result)
            
            avg_enhanced_time = np.mean(enhanced_times)
            print(f"      Enhanced processing: {avg_enhanced_time:.3f}ms avg")
            
            # Compare results
            changes = 0
            for std, enh in zip(standard_results, enhanced_results):
                if std != enh:
                    changes += 1
            
            change_rate = changes / len(test_descriptions) * 100
            print(f"      Text changes: {changes}/{len(test_descriptions)} ({change_rate:.1f}%)")
            
            return {
                'available': True,
                'standard_time_ms': avg_standard_time,
                'enhanced_time_ms': avg_enhanced_time,
                'change_rate_percent': change_rate,
                'test_descriptions': len(test_descriptions)
            }
            
        except Exception as e:
            print(f"      ⚠️ Enhanced processing not available: {e}")
            return {
                'available': False,
                'standard_time_ms': avg_standard_time,
                'test_descriptions': len(test_descriptions),
                'error': str(e)
            }
        
    except Exception as e:
        print(f"   ❌ Text processing error: {e}")
        return {'available': False, 'error': str(e)}

def run_quick_performance_test():
    """Run quick performance test"""
    print("🚀 QUICK PERFORMANCE TEST")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test 1: Database Performance
    db_results = test_database_performance()
    
    # Test 2: Cache System
    cache_results = test_cache_system()
    
    # Test 3: Enhanced Text Processing
    text_results = test_enhanced_text_processing()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Generate summary
    print(f"\n📋 QUICK TEST SUMMARY")
    print("=" * 50)
    print(f"⏱️ Total test time: {total_time:.1f} seconds")
    
    # Database summary
    if db_results:
        fast_queries = sum(1 for r in db_results.values() if r['is_fast'])
        total_queries = len(db_results)
        print(f"🗄️ Database: {fast_queries}/{total_queries} fast queries ({fast_queries/total_queries*100:.0f}%)")
    
    # Cache summary
    if cache_results.get('available', False):
        print(f"💾 Cache: Available, {cache_results['hit_rate_percent']:.0f}% hit rate")
    else:
        print(f"💾 Cache: Not available")
    
    # Text processing summary
    if text_results.get('available', False):
        if 'enhanced_time_ms' in text_results:
            print(f"📝 Text Processing: Enhanced available, {text_results['change_rate_percent']:.0f}% improvements")
        else:
            print(f"📝 Text Processing: Standard only")
    else:
        print(f"📝 Text Processing: Not available")
    
    # Overall assessment
    components_working = 0
    total_components = 3
    
    if db_results:
        components_working += 1
    if cache_results.get('available', False):
        components_working += 1
    if text_results.get('available', False):
        components_working += 1
    
    overall_score = components_working / total_components * 100
    
    print(f"\n🎯 Overall Status: {components_working}/{total_components} components working ({overall_score:.0f}%)")
    
    if overall_score >= 80:
        print("🎉 EXCELLENT - Performance improvements are working well!")
    elif overall_score >= 60:
        print("✅ GOOD - Most improvements are working")
    elif overall_score >= 40:
        print("⚠️ FAIR - Some improvements need attention")
    else:
        print("❌ POOR - Significant issues need to be addressed")
    
    return {
        'database_performance': db_results,
        'cache_performance': cache_results,
        'text_processing': text_results,
        'overall_score': overall_score,
        'total_time_seconds': total_time
    }

if __name__ == "__main__":
    results = run_quick_performance_test()
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f"quick_test_results_{timestamp}.json"
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📁 Results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")
    
    print("\n✅ Quick performance test completed!")
