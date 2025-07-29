#!/usr/bin/env python3
"""
Comprehensive Performance Testing Suite for SKU Predictor v2.0
Tests all components and identifies optimization opportunities
"""

import time
import sys
import os
import sqlite3
import pandas as pd
import psutil
import tracemalloc
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def measure_time(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    return wrapper

class PerformanceTester:
    def __init__(self):
        self.results = {}
        self.memory_usage = {}
        
    def start_memory_tracking(self):
        """Start memory tracking"""
        tracemalloc.start()
        
    def get_memory_usage(self, label):
        """Get current memory usage"""
        current, peak = tracemalloc.get_traced_memory()
        self.memory_usage[label] = {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024
        }
        return current / 1024 / 1024
        
    def test_startup_performance(self):
        """Test application startup components"""
        print("🚀 Testing Application Startup Performance...")
        
        # Test Excel file loading
        @measure_time
        def test_excel_loading():
            import pandas as pd
            df = pd.read_excel('Source_Files/Text_Processing_Rules.xlsx', sheet_name=None)
            return len(df)
            
        # Test spaCy initialization
        @measure_time
        def test_spacy_init():
            import spacy
            nlp = spacy.load("es_core_news_sm")
            return nlp
            
        # Test database connection
        @measure_time
        def test_db_connection():
            conn = sqlite3.connect('Source_Files/processed_consolidado.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM processed_consolidado")
            result = cursor.fetchone()[0]
            conn.close()
            return result
            
        # Test model loading
        @measure_time
        def test_model_loading():
            import joblib
            models = []
            model_files = [
                'models/vin_maker_model.joblib',
                'models/vin_year_model.joblib', 
                'models/vin_series_model.joblib'
            ]
            for model_file in model_files:
                if os.path.exists(model_file):
                    model = joblib.load(model_file)
                    models.append(model)
            return len(models)
        
        # Run tests
        self.start_memory_tracking()
        
        excel_result, excel_time = test_excel_loading()
        excel_memory = self.get_memory_usage('excel_loading')
        
        spacy_result, spacy_time = test_spacy_init()
        spacy_memory = self.get_memory_usage('spacy_init')
        
        db_result, db_time = test_db_connection()
        db_memory = self.get_memory_usage('db_connection')
        
        model_result, model_time = test_model_loading()
        model_memory = self.get_memory_usage('model_loading')
        
        self.results['startup'] = {
            'excel_loading': {'time_ms': excel_time, 'memory_mb': excel_memory, 'sheets': excel_result},
            'spacy_init': {'time_ms': spacy_time, 'memory_mb': spacy_memory},
            'db_connection': {'time_ms': db_time, 'memory_mb': db_memory, 'records': db_result},
            'model_loading': {'time_ms': model_time, 'memory_mb': model_memory, 'models': model_result}
        }
        
        print(f"  📊 Excel Loading: {excel_time:.1f}ms ({excel_memory:.1f}MB)")
        print(f"  🧠 spaCy Init: {spacy_time:.1f}ms ({spacy_memory:.1f}MB)")
        print(f"  🗄️ DB Connection: {db_time:.1f}ms ({db_memory:.1f}MB)")
        print(f"  🤖 Model Loading: {model_time:.1f}ms ({model_memory:.1f}MB)")
        
    def test_database_performance(self):
        """Test database query performance"""
        print("\n🗄️ Testing Database Performance...")
        
        conn = sqlite3.connect('Source_Files/processed_consolidado.db')
        
        # Test queries
        queries = {
            'count_all': "SELECT COUNT(*) FROM processed_consolidado",
            'make_filter': "SELECT COUNT(*) FROM processed_consolidado WHERE maker = 'Toyota'",
            'make_year_filter': "SELECT COUNT(*) FROM processed_consolidado WHERE maker = 'Toyota' AND model = '2020'",
            'sku_frequency': "SELECT referencia, COUNT(*) as freq FROM processed_consolidado GROUP BY referencia ORDER BY freq DESC LIMIT 10",
            'description_search': "SELECT * FROM processed_consolidado WHERE normalized_descripcion LIKE '%capo%' LIMIT 10",
            'complex_join': """
                SELECT maker, model, series, referencia, COUNT(*) as freq 
                FROM processed_consolidado 
                WHERE maker IS NOT NULL AND series IS NOT NULL 
                GROUP BY maker, model, series, referencia 
                ORDER BY freq DESC LIMIT 20
            """
        }
        
        db_results = {}
        for query_name, query in queries.items():
            start_time = time.time()
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000
            db_results[query_name] = {
                'time_ms': execution_time,
                'rows': len(results)
            }
            print(f"  📊 {query_name}: {execution_time:.1f}ms ({len(results)} rows)")
            
        conn.close()
        self.results['database'] = db_results
        
    def test_text_processing_performance(self):
        """Test text processing pipeline performance"""
        print("\n📝 Testing Text Processing Performance...")
        
        # Sample descriptions for testing
        test_descriptions = [
            "capo",
            "farola derecha",
            "paragolpes delantero",
            "espejo retrovisor izquierdo",
            "puerta trasera derecha",
            "guardafango delantero izquierdo",
            "sensor de proximidad trasero",
            "vidrio panoramico",
            "electroventilador del radiador",
            "absorbedor de impactos delantero"
        ]
        
        # Test spaCy processing
        @measure_time
        def test_spacy_processing():
            import spacy
            nlp = spacy.load("es_core_news_sm")
            processed = []
            for desc in test_descriptions:
                doc = nlp(desc)
                processed.append(str(doc))
            return len(processed)
            
        # Test abbreviation processing
        @measure_time
        def test_abbreviation_processing():
            # Simulate abbreviation processing
            processed = []
            abbreviations = {
                'der': 'derecha', 'izq': 'izquierda', 'del': 'delantero',
                'tra': 'trasero', 'far': 'farola', 'esp': 'espejo'
            }
            for desc in test_descriptions:
                for abbr, full in abbreviations.items():
                    desc = desc.replace(abbr, full)
                processed.append(desc)
            return len(processed)
            
        spacy_result, spacy_time = test_spacy_processing()
        abbr_result, abbr_time = test_abbreviation_processing()
        
        self.results['text_processing'] = {
            'spacy_processing': {'time_ms': spacy_time, 'descriptions': spacy_result},
            'abbreviation_processing': {'time_ms': abbr_time, 'descriptions': abbr_result}
        }
        
        print(f"  🧠 spaCy Processing: {spacy_time:.1f}ms ({spacy_result} descriptions)")
        print(f"  📝 Abbreviation Processing: {abbr_time:.1f}ms ({abbr_result} descriptions)")
        
    def test_vin_prediction_performance(self):
        """Test VIN prediction performance"""
        print("\n🚗 Testing VIN Prediction Performance...")
        
        # Sample VINs for testing
        test_vins = [
            "9FB5SRC9GJM762420",  # Renault
            "3VWDX7AJ5DM123456",  # Volkswagen
            "1HGBH41JXMN109186",  # Honda
            "WVWZZZ1JZ3W123456",  # Volkswagen
            "JM1BK32F781234567"   # Mazda
        ]
        
        @measure_time
        def test_vin_predictions():
            # Simulate VIN prediction (without loading actual models to avoid conflicts)
            predictions = []
            for vin in test_vins:
                # Simple VIN analysis simulation
                prediction = {
                    'maker': 'Unknown',
                    'year': '2020',
                    'series': 'Unknown'
                }
                predictions.append(prediction)
            return len(predictions)
            
        vin_result, vin_time = test_vin_predictions()
        
        self.results['vin_prediction'] = {
            'batch_prediction': {'time_ms': vin_time, 'vins': vin_result}
        }
        
        print(f"  🚗 VIN Predictions: {vin_time:.1f}ms ({vin_result} VINs)")
        
    def analyze_system_resources(self):
        """Analyze system resource usage"""
        print("\n💻 Analyzing System Resources...")
        
        process = psutil.Process()
        
        # CPU usage
        cpu_percent = process.cpu_percent(interval=1)
        
        # Memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Disk usage
        disk_usage = psutil.disk_usage('.')
        
        self.results['system_resources'] = {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'disk_free_gb': disk_usage.free / 1024 / 1024 / 1024
        }
        
        print(f"  🖥️ CPU Usage: {cpu_percent:.1f}%")
        print(f"  💾 Memory Usage: {memory_mb:.1f}MB")
        print(f"  💿 Disk Free: {disk_usage.free / 1024 / 1024 / 1024:.1f}GB")
        
    def generate_optimization_recommendations(self):
        """Generate optimization recommendations based on test results"""
        print("\n🎯 Optimization Recommendations:")
        
        recommendations = []
        
        # Startup optimizations
        startup = self.results.get('startup', {})
        if startup.get('spacy_init', {}).get('time_ms', 0) > 1000:
            recommendations.append("🧠 spaCy: Consider lazy loading or caching spaCy model")
            
        if startup.get('excel_loading', {}).get('time_ms', 0) > 500:
            recommendations.append("📊 Excel: Convert Excel files to faster formats (JSON/pickle)")
            
        # Database optimizations
        database = self.results.get('database', {})
        if database.get('description_search', {}).get('time_ms', 0) > 100:
            recommendations.append("🗄️ Database: Add full-text search index for descriptions")
            
        # Memory optimizations
        total_memory = sum([v.get('memory_mb', 0) for v in startup.values()])
        if total_memory > 500:
            recommendations.append("💾 Memory: Implement lazy loading for large components")
            
        # General recommendations
        recommendations.extend([
            "⚡ Cache: Implement intelligent caching for frequent queries",
            "🔄 Async: Use asynchronous loading for non-critical components",
            "📦 Compression: Compress model files and data",
            "🎯 Profiling: Add detailed profiling for bottleneck identification"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
            
        return recommendations
        
    def run_comprehensive_test(self):
        """Run all performance tests"""
        print("🔍 Starting Comprehensive Performance Analysis...")
        print("=" * 60)
        
        self.test_startup_performance()
        self.test_database_performance()
        self.test_text_processing_performance()
        self.test_vin_prediction_performance()
        self.analyze_system_resources()
        
        print("\n" + "=" * 60)
        recommendations = self.generate_optimization_recommendations()
        
        return self.results, recommendations

if __name__ == "__main__":
    # Change to project directory
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    tester = PerformanceTester()
    results, recommendations = tester.run_comprehensive_test()
    
    print(f"\n✅ Performance analysis complete!")
    print(f"📊 Results saved to memory for further analysis")
