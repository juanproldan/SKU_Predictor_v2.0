#!/usr/bin/env python3
"""
Test the performance optimizations
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_startup_performance():
    """Test optimized startup performance"""
    print("🚀 Testing Optimized Startup Performance...")
    
    # Test optimized components
    start_time = time.time()
    
    # Test optimized data loader
    from utils.optimized_startup import get_data_loader
    data_loader = get_data_loader()
    
    # Test Excel loading with caching
    excel_file = "Source_Files/Text_Processing_Rules.xlsx"
    if os.path.exists(excel_file):
        rules = data_loader.load_text_processing_rules_optimized(excel_file)
        print(f"  📊 Optimized Excel loading: {len(rules)} rule sets loaded")
    
    # Test spaCy lazy loading
    from utils.optimized_startup import get_spacy_loader
    spacy_loader = get_spacy_loader()
    spacy_loader.start_loading()
    print("  🧠 spaCy loading started in background")
    
    # Test optimized database
    from utils.optimized_database import get_optimized_database
    db = get_optimized_database()
    print("  🗄️ Optimized database initialized")
    
    startup_time = (time.time() - start_time) * 1000
    print(f"  ⏱️ Total optimized startup: {startup_time:.1f}ms")
    
    return startup_time

def test_database_performance():
    """Test optimized database performance"""
    print("\n🗄️ Testing Optimized Database Performance...")
    
    from utils.optimized_database import get_optimized_database, benchmark_database_performance
    
    # Run benchmark
    avg_time = benchmark_database_performance()
    
    # Test cache performance
    db = get_optimized_database()
    cache_stats = db.get_cache_stats()
    
    print(f"  📊 Cache performance:")
    print(f"    Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    print(f"    Cached queries: {cache_stats['cached_queries']}")
    
    return avg_time

def test_text_processing_performance():
    """Test fast text processing"""
    print("\n📝 Testing Fast Text Processing...")
    
    # Test descriptions
    test_descriptions = [
        "capo del vehiculo",
        "farola der con antiniebla", 
        "paragolpes del c/ sensor",
        "espejo retrovisor izq electrico",
        "puerta tra der c/ vidrio"
    ]
    
    # Load rules for fast processor
    from utils.optimized_startup import get_data_loader, get_text_processor
    data_loader = get_data_loader()
    
    excel_file = "Source_Files/Text_Processing_Rules.xlsx"
    if os.path.exists(excel_file):
        rules = data_loader.load_text_processing_rules_optimized(excel_file)
        fast_processor = get_text_processor(rules)
        
        if fast_processor:
            total_time = 0
            for desc in test_descriptions:
                start_time = time.time()
                processed = fast_processor.process_fast(desc)
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000
                total_time += processing_time
                
                print(f"    '{desc}' → '{processed}' ({processing_time:.3f}ms)")
            
            avg_time = total_time / len(test_descriptions)
            print(f"  📊 Average processing time: {avg_time:.3f}ms")
            return avg_time
    
    print("  ⚠️ Fast text processor not available")
    return 0

def test_spacy_performance():
    """Test spaCy loading performance"""
    print("\n🧠 Testing spaCy Performance...")
    
    from utils.optimized_startup import get_spacy_loader
    spacy_loader = get_spacy_loader()
    
    # Check if already loaded
    if spacy_loader.is_ready():
        print("  ✅ spaCy already loaded")
        
        # Test processing
        nlp = spacy_loader.get_nlp()
        if nlp:
            test_text = "farola derecha del vehiculo"
            start_time = time.time()
            doc = nlp(test_text)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000
            print(f"  📝 spaCy processing: '{test_text}' ({processing_time:.1f}ms)")
            return processing_time
    else:
        print("  ⏳ spaCy still loading...")
        # Wait for it to load
        nlp = spacy_loader.get_nlp(timeout=10)
        if nlp:
            print("  ✅ spaCy loaded successfully")
            return 0
        else:
            print("  ❌ spaCy loading failed")
            return -1

def compare_with_baseline():
    """Compare optimized vs baseline performance"""
    print("\n📊 Performance Comparison Summary:")
    
    # Run optimized tests
    startup_time = test_startup_performance()
    db_time = test_database_performance()
    text_time = test_text_processing_performance()
    spacy_time = test_spacy_performance()
    
    # Baseline estimates from previous tests
    baseline_startup = 21000  # 21 seconds from comprehensive test
    baseline_db = 1200  # 1.2 seconds average query time
    baseline_text = 0.1  # Very fast but no caching
    baseline_spacy = 16000  # 16 seconds initialization
    
    print(f"\n🎯 Performance Improvements:")
    
    if startup_time > 0:
        startup_improvement = ((baseline_startup - startup_time) / baseline_startup) * 100
        print(f"  🚀 Startup: {startup_time:.1f}ms vs {baseline_startup}ms baseline ({startup_improvement:.1f}% improvement)")
    
    if db_time > 0:
        db_improvement = ((baseline_db - db_time) / baseline_db) * 100
        print(f"  🗄️ Database: {db_time:.1f}ms vs {baseline_db}ms baseline ({db_improvement:.1f}% improvement)")
    
    if text_time > 0:
        print(f"  📝 Text Processing: {text_time:.3f}ms (optimized with caching)")
    
    if spacy_time >= 0:
        print(f"  🧠 spaCy: Background loading vs {baseline_spacy}ms baseline (async improvement)")

def run_optimization_tests():
    """Run all optimization tests"""
    print("🔍 Testing Performance Optimizations...")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    # Run tests
    compare_with_baseline()
    
    print("\n" + "=" * 60)
    print("✅ Optimization testing complete!")

if __name__ == "__main__":
    run_optimization_tests()
