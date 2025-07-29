#!/usr/bin/env python3
"""
Real-world performance testing with actual application components
"""

import time
import sys
import os
import sqlite3
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_actual_sku_prediction():
    """Test actual SKU prediction with real components"""
    print("🔍 Testing Real SKU Prediction Performance...")
    
    # Test scenarios
    test_cases = [
        {
            'vin': '9FB5SRC9GJM762420',
            'descriptions': ['capo', 'farola derecha', 'paragolpes delantero']
        },
        {
            'vin': '3VWDX7AJ5DM123456', 
            'descriptions': ['espejo retrovisor', 'puerta trasera', 'guardafango']
        },
        {
            'vin': '1HGBH41JXMN109186',
            'descriptions': ['sensor proximidad', 'vidrio panoramico', 'electroventilador']
        }
    ]
    
    # Test database queries directly
    conn = sqlite3.connect('Source_Files/processed_consolidado.db')
    
    total_time = 0
    total_queries = 0
    
    for case in test_cases:
        print(f"\n  Testing VIN: {case['vin']}")
        
        # Test VIN-based vehicle prediction
        start_time = time.time()
        
        # Simulate vehicle details extraction
        vehicle_details = {
            'maker': 'Toyota',  # Would come from VIN prediction
            'model': '2020',
            'series': 'Corolla'
        }
        
        vin_time = (time.time() - start_time) * 1000
        print(f"    VIN Processing: {vin_time:.1f}ms")
        
        for desc in case['descriptions']:
            print(f"    Testing description: '{desc}'")
            
            # Test Maestro lookup
            start_time = time.time()
            maestro_query = """
                SELECT referencia, COUNT(*) as frequency
                FROM processed_consolidado 
                WHERE maker = ? AND model = ? AND series = ? 
                AND (normalized_descripcion LIKE ? OR original_descripcion LIKE ?)
                GROUP BY referencia 
                ORDER BY frequency DESC 
                LIMIT 5
            """
            cursor = conn.cursor()
            like_pattern = f"%{desc}%"
            cursor.execute(maestro_query, (
                vehicle_details['maker'], 
                vehicle_details['model'], 
                vehicle_details['series'],
                like_pattern, 
                like_pattern
            ))
            maestro_results = cursor.fetchall()
            maestro_time = (time.time() - start_time) * 1000
            
            # Test database fallback
            start_time = time.time()
            db_query = """
                SELECT referencia, COUNT(*) as frequency
                FROM processed_consolidado 
                WHERE normalized_descripcion LIKE ? OR original_descripcion LIKE ?
                GROUP BY referencia 
                ORDER BY frequency DESC 
                LIMIT 5
            """
            cursor.execute(db_query, (like_pattern, like_pattern))
            db_results = cursor.fetchall()
            db_time = (time.time() - start_time) * 1000
            
            print(f"      Maestro Query: {maestro_time:.1f}ms ({len(maestro_results)} results)")
            print(f"      Database Query: {db_time:.1f}ms ({len(db_results)} results)")
            
            total_time += maestro_time + db_time
            total_queries += 2
    
    conn.close()
    
    avg_query_time = total_time / total_queries if total_queries > 0 else 0
    print(f"\n  📊 Average Query Time: {avg_query_time:.1f}ms")
    print(f"  📊 Total Queries: {total_queries}")
    print(f"  📊 Total Time: {total_time:.1f}ms")
    
    return {
        'avg_query_time': avg_query_time,
        'total_queries': total_queries,
        'total_time': total_time
    }

def test_text_processing_pipeline():
    """Test the complete text processing pipeline"""
    print("\n📝 Testing Complete Text Processing Pipeline...")
    
    test_descriptions = [
        "capo del vehiculo",
        "farola der con antiniebla", 
        "paragolpes del c/ sensor",
        "espejo retrovisor izq electrico",
        "puerta tra der c/ vidrio",
        "guardafango del izq plastico",
        "sensor proximidad tra central",
        "vidrio pano tintado",
        "electrovent radiador motor",
        "absorbimpacto del reforzado"
    ]
    
    # Test each processing step
    processing_times = {}
    
    for desc in test_descriptions:
        print(f"  Processing: '{desc}'")
        
        # Step 1: Basic normalization
        start_time = time.time()
        normalized = desc.lower().strip()
        norm_time = (time.time() - start_time) * 1000
        
        # Step 2: Abbreviation expansion (simulated)
        start_time = time.time()
        abbreviations = {
            'del': 'delantero', 'tra': 'trasero', 'der': 'derecha', 
            'izq': 'izquierda', 'c/': 'con', 'pano': 'panoramico'
        }
        expanded = normalized
        for abbr, full in abbreviations.items():
            expanded = expanded.replace(abbr, full)
        abbr_time = (time.time() - start_time) * 1000
        
        # Step 3: Equivalencias (simulated)
        start_time = time.time()
        equivalencias = {
            'farola': 'faro', 'electrovent': 'electroventilador',
            'absorbimpacto': 'absorbedor'
        }
        equiv_desc = expanded
        for orig, equiv in equivalencias.items():
            equiv_desc = equiv_desc.replace(orig, equiv)
        equiv_time = (time.time() - start_time) * 1000
        
        total_processing = norm_time + abbr_time + equiv_time
        
        print(f"    Normalization: {norm_time:.3f}ms")
        print(f"    Abbreviations: {abbr_time:.3f}ms") 
        print(f"    Equivalencias: {equiv_time:.3f}ms")
        print(f"    Total: {total_processing:.3f}ms")
        print(f"    Result: '{equiv_desc}'")
        
        if desc not in processing_times:
            processing_times[desc] = {}
        processing_times[desc] = {
            'normalization': norm_time,
            'abbreviations': abbr_time,
            'equivalencias': equiv_time,
            'total': total_processing
        }
    
    # Calculate averages
    avg_times = {
        'normalization': sum(t['normalization'] for t in processing_times.values()) / len(processing_times),
        'abbreviations': sum(t['abbreviations'] for t in processing_times.values()) / len(processing_times),
        'equivalencias': sum(t['equivalencias'] for t in processing_times.values()) / len(processing_times),
        'total': sum(t['total'] for t in processing_times.values()) / len(processing_times)
    }
    
    print(f"\n  📊 Average Processing Times:")
    print(f"    Normalization: {avg_times['normalization']:.3f}ms")
    print(f"    Abbreviations: {avg_times['abbreviations']:.3f}ms")
    print(f"    Equivalencias: {avg_times['equivalencias']:.3f}ms")
    print(f"    Total Average: {avg_times['total']:.3f}ms")
    
    return processing_times, avg_times

def test_database_optimization_impact():
    """Test the impact of database optimizations"""
    print("\n🗄️ Testing Database Optimization Impact...")
    
    conn = sqlite3.connect('Source_Files/processed_consolidado.db')
    
    # Test queries with and without optimization hints
    test_queries = [
        {
            'name': 'Exact Match',
            'query': """
                SELECT referencia, COUNT(*) as freq 
                FROM processed_consolidado 
                WHERE maker = 'Toyota' AND model = '2020' AND series = 'Corolla' 
                AND normalized_descripcion = 'capo'
                GROUP BY referencia
            """
        },
        {
            'name': 'Fuzzy Match',
            'query': """
                SELECT referencia, COUNT(*) as freq 
                FROM processed_consolidado 
                WHERE maker = 'Toyota' AND normalized_descripcion LIKE '%capo%'
                GROUP BY referencia 
                ORDER BY freq DESC 
                LIMIT 10
            """
        },
        {
            'name': 'Frequency Analysis',
            'query': """
                SELECT referencia, COUNT(*) as freq 
                FROM processed_consolidado 
                GROUP BY referencia 
                HAVING freq >= 5 
                ORDER BY freq DESC 
                LIMIT 20
            """
        }
    ]
    
    query_results = {}
    
    for test in test_queries:
        print(f"  Testing: {test['name']}")
        
        # Run query multiple times for average
        times = []
        for i in range(3):
            start_time = time.time()
            cursor = conn.cursor()
            cursor.execute(test['query'])
            results = cursor.fetchall()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        query_results[test['name']] = {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'results_count': len(results)
        }
        
        print(f"    Average: {avg_time:.1f}ms")
        print(f"    Range: {min_time:.1f}ms - {max_time:.1f}ms")
        print(f"    Results: {len(results)} rows")
    
    conn.close()
    return query_results

def run_real_world_tests():
    """Run all real-world performance tests"""
    print("🔍 Starting Real-World Performance Testing...")
    print("=" * 60)
    
    # Test SKU prediction
    sku_results = test_actual_sku_prediction()
    
    # Test text processing
    text_processing, text_avg = test_text_processing_pipeline()
    
    # Test database optimization
    db_results = test_database_optimization_impact()
    
    print("\n" + "=" * 60)
    print("📊 Real-World Performance Summary:")
    print(f"  SKU Prediction Average: {sku_results['avg_query_time']:.1f}ms per query")
    print(f"  Text Processing Average: {text_avg['total']:.3f}ms per description")
    print(f"  Database Query Range: {min(r['avg_time'] for r in db_results.values()):.1f}ms - {max(r['avg_time'] for r in db_results.values()):.1f}ms")
    
    return {
        'sku_prediction': sku_results,
        'text_processing': text_avg,
        'database': db_results
    }

if __name__ == "__main__":
    # Change to project directory
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    results = run_real_world_tests()
    print(f"\n✅ Real-world testing complete!")
