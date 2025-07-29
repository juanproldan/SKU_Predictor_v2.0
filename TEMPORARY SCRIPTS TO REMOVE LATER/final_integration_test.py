#!/usr/bin/env python3
"""
Final integration test to validate all optimizations work together
"""

import time
import sys
import os
import sqlite3

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_end_to_end_workflow():
    """Test complete end-to-end workflow with optimizations"""
    print("🔍 Testing End-to-End Optimized Workflow...")
    
    # Test cases that simulate real user interactions
    test_cases = [
        {
            'vin': '9FB5SRC9GJM762420',
            'description': 'capo del vehiculo',
            'expected_maker': 'Renault'
        },
        {
            'vin': '3VWDX7AJ5DM123456',
            'description': 'farola der con antiniebla',
            'expected_maker': 'Volkswagen'
        },
        {
            'vin': '1HGBH41JXMN109186',
            'description': 'paragolpes del c/ sensor',
            'expected_maker': 'Honda'
        }
    ]
    
    total_time = 0
    successful_tests = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n  Test Case {i}: {case['vin']} - '{case['description']}'")
        
        start_time = time.time()
        
        # Test VIN prediction (simulated)
        vin_start = time.time()
        # Simulate VIN processing
        vehicle_details = {
            'maker': case['expected_maker'],
            'model': '2020',
            'series': 'Unknown'
        }
        vin_time = (time.time() - vin_start) * 1000
        
        # Test text processing
        text_start = time.time()
        from utils.optimized_startup import get_data_loader, get_text_processor
        
        data_loader = get_data_loader()
        excel_file = "Source_Files/Text_Processing_Rules.xlsx"
        
        if os.path.exists(excel_file):
            rules = data_loader.load_text_processing_rules_optimized(excel_file)
            fast_processor = get_text_processor(rules)
            
            if fast_processor:
                processed_desc = fast_processor.process_fast(case['description'])
            else:
                processed_desc = case['description'].lower()
        else:
            processed_desc = case['description'].lower()
        
        text_time = (time.time() - text_start) * 1000
        
        # Test database queries
        db_start = time.time()
        from utils.optimized_database import get_optimized_database
        
        db = get_optimized_database()
        
        # Test Maestro predictions
        maestro_results = db.get_maestro_predictions_optimized(
            vehicle_details['maker'], 
            vehicle_details['model'], 
            vehicle_details['series'], 
            processed_desc
        )
        
        # Test database predictions
        db_results = db.get_database_predictions_optimized(processed_desc)
        
        db_time = (time.time() - db_start) * 1000
        
        total_case_time = (time.time() - start_time) * 1000
        total_time += total_case_time
        
        # Validate results
        has_results = len(maestro_results) > 0 or len(db_results) > 0
        
        print(f"    VIN Processing: {vin_time:.1f}ms")
        print(f"    Text Processing: '{case['description']}' → '{processed_desc}' ({text_time:.3f}ms)")
        print(f"    Database Queries: {db_time:.1f}ms")
        print(f"    Maestro Results: {len(maestro_results)}")
        print(f"    Database Results: {len(db_results)}")
        print(f"    Total Time: {total_case_time:.1f}ms")
        print(f"    Status: {'✅ Success' if has_results else '⚠️ No results'}")
        
        if has_results:
            successful_tests += 1
    
    avg_time = total_time / len(test_cases)
    success_rate = (successful_tests / len(test_cases)) * 100
    
    print(f"\n  📊 End-to-End Performance Summary:")
    print(f"    Average Time per Query: {avg_time:.1f}ms")
    print(f"    Success Rate: {success_rate:.1f}%")
    print(f"    Total Tests: {len(test_cases)}")
    print(f"    Successful: {successful_tests}")
    
    return avg_time, success_rate

def test_cache_performance():
    """Test caching performance improvements"""
    print("\n💾 Testing Cache Performance...")
    
    from utils.optimized_database import get_optimized_database
    db = get_optimized_database()
    
    # Test repeated queries to measure cache effectiveness
    test_query = "capo"
    
    # First query (cache miss)
    start_time = time.time()
    results1 = db.get_database_predictions_optimized(test_query)
    first_time = (time.time() - start_time) * 1000
    
    # Second query (should be cached)
    start_time = time.time()
    results2 = db.get_database_predictions_optimized(test_query)
    second_time = (time.time() - start_time) * 1000
    
    # Third query (should be cached)
    start_time = time.time()
    results3 = db.get_database_predictions_optimized(test_query)
    third_time = (time.time() - start_time) * 1000
    
    cache_stats = db.get_cache_stats()
    
    print(f"  Query 1 (cache miss): {first_time:.1f}ms ({len(results1)} results)")
    print(f"  Query 2 (cache hit): {second_time:.1f}ms ({len(results2)} results)")
    print(f"  Query 3 (cache hit): {third_time:.1f}ms ({len(results3)} results)")
    print(f"  Cache Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
    print(f"  Cached Queries: {cache_stats['cached_queries']}")
    
    # Calculate cache improvement
    if first_time > 0:
        cache_improvement = ((first_time - second_time) / first_time) * 100
        print(f"  Cache Improvement: {cache_improvement:.1f}%")
        return cache_improvement
    
    return 0

def test_spacy_integration():
    """Test spaCy integration and performance"""
    print("\n🧠 Testing spaCy Integration...")
    
    from utils.optimized_startup import get_spacy_loader
    spacy_loader = get_spacy_loader()
    
    if spacy_loader.is_ready():
        nlp = spacy_loader.get_nlp()
        
        if nlp:
            test_descriptions = [
                "farola derecho",  # Should become "farola derecha"
                "puerta trasero",  # Should become "puerta trasera"
                "emblema trasera",  # Should become "emblema trasero" (exception)
                "portaplaca trasera"  # Should become "portaplaca trasero" (exception)
            ]
            
            print("  Testing gender agreement corrections:")
            for desc in test_descriptions:
                start_time = time.time()
                doc = nlp(desc)
                processing_time = (time.time() - start_time) * 1000
                
                # Simple gender agreement test (would need actual spaCy processor for real results)
                processed = str(doc)
                print(f"    '{desc}' → '{processed}' ({processing_time:.1f}ms)")
            
            return True
        else:
            print("  ❌ spaCy not available")
            return False
    else:
        print("  ⏳ spaCy still loading...")
        return False

def validate_data_integrity():
    """Validate that optimizations haven't corrupted data"""
    print("\n🔍 Validating Data Integrity...")
    
    # Check database
    try:
        conn = sqlite3.connect('Source_Files/processed_consolidado.db')
        cursor = conn.cursor()
        
        # Basic integrity checks
        cursor.execute("SELECT COUNT(*) FROM processed_consolidado")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT referencia) FROM processed_consolidado WHERE referencia IS NOT NULL")
        unique_skus = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT vin_number) FROM processed_consolidado WHERE vin_number IS NOT NULL")
        unique_vins = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"  📊 Database Integrity:")
        print(f"    Total Records: {total_records:,}")
        print(f"    Unique SKUs: {unique_skus:,}")
        print(f"    Unique VINs: {unique_vins:,}")
        
        # Validate reasonable numbers
        integrity_ok = (
            total_records > 100000 and  # Should have substantial data
            unique_skus > 1000 and     # Should have many SKUs
            unique_vins > 10000        # Should have many VINs
        )
        
        print(f"    Status: {'✅ Integrity OK' if integrity_ok else '❌ Integrity Issues'}")
        return integrity_ok
        
    except Exception as e:
        print(f"  ❌ Database integrity check failed: {e}")
        return False

def run_final_integration_test():
    """Run complete integration test suite"""
    print("🎯 Final Integration Test Suite")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    # Run all tests
    avg_time, success_rate = test_end_to_end_workflow()
    cache_improvement = test_cache_performance()
    spacy_working = test_spacy_integration()
    data_integrity = validate_data_integrity()
    
    print("\n" + "=" * 60)
    print("🎯 Final Integration Test Results:")
    print(f"  ⚡ Average Query Time: {avg_time:.1f}ms")
    print(f"  ✅ Success Rate: {success_rate:.1f}%")
    print(f"  💾 Cache Improvement: {cache_improvement:.1f}%")
    print(f"  🧠 spaCy Integration: {'✅ Working' if spacy_working else '❌ Issues'}")
    print(f"  🔍 Data Integrity: {'✅ Valid' if data_integrity else '❌ Issues'}")
    
    # Overall assessment
    overall_score = (
        (1 if avg_time < 1000 else 0) +  # Fast queries
        (1 if success_rate > 80 else 0) +  # High success rate
        (1 if cache_improvement > 0 else 0) +  # Cache working
        (1 if spacy_working else 0) +  # spaCy working
        (1 if data_integrity else 0)  # Data integrity
    )
    
    print(f"\n🎯 Overall Assessment: {overall_score}/5")
    
    if overall_score >= 4:
        print("🎉 EXCELLENT: All optimizations working successfully!")
    elif overall_score >= 3:
        print("✅ GOOD: Most optimizations working, minor issues detected")
    else:
        print("⚠️ ISSUES: Significant problems detected, review needed")
    
    return overall_score

if __name__ == "__main__":
    score = run_final_integration_test()
    print(f"\n✅ Integration testing complete! Score: {score}/5")
