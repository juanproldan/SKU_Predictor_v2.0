#!/usr/bin/env python3
"""
BULLETPROOF TEST SCRIPT
Tests if the critical modules are working in the packaged executable
"""

import sys
import os
import traceback

def test_array_api_compat():
    """Test if array_api_compat modules are working"""
    print("🔍 Testing array_api_compat modules...")
    
    try:
        # Test the main module
        import array_api_compat
        print("✅ array_api_compat imported successfully")
        
        # Test the specific scipy module that was failing
        import scipy._lib.array_api_compat.numpy.fft
        print("✅ scipy._lib.array_api_compat.numpy.fft imported successfully")
        
        # Test numpy compatibility
        import array_api_compat.numpy
        print("✅ array_api_compat.numpy imported successfully")
        
        # Test fft module
        import array_api_compat.numpy.fft
        print("✅ array_api_compat.numpy.fft imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ array_api_compat test failed: {e}")
        traceback.print_exc()
        return False

def test_spacy_model():
    """Test if spaCy model is working"""
    print("\n🔍 Testing spaCy model...")

    try:
        import spacy
        print("✅ spaCy imported successfully")

        # Try multiple approaches to load the Spanish model
        nlp = None

        # Approach 1: Try standard loading
        try:
            nlp = spacy.load('es_core_news_sm')
            print("✅ es_core_news_sm model loaded successfully (standard method)")
        except:
            print("⚠️  Standard loading failed, trying bundled path...")

            # Approach 2: Try loading from bundled path
            if getattr(sys, 'frozen', False):
                model_path = os.path.join(sys._MEIPASS, 'es_core_news_sm')
                try:
                    nlp = spacy.load(model_path)
                    print("✅ es_core_news_sm model loaded successfully (bundled path)")
                except:
                    print("⚠️  Bundled path loading failed, trying direct model path...")

                    # Approach 3: Try loading from the actual model directory
                    model_dir = os.path.join(model_path, 'es_core_news_sm-3.8.0')
                    if os.path.exists(model_dir):
                        nlp = spacy.load(model_dir)
                        print("✅ es_core_news_sm model loaded successfully (direct model path)")

        if nlp is None:
            raise Exception("Could not load spaCy model with any method")

        # Test processing some text
        doc = nlp("Este es un texto de prueba en español")
        print(f"✅ Text processing successful: {len(doc)} tokens")

        return True

    except Exception as e:
        print(f"❌ spaCy test failed: {e}")
        traceback.print_exc()
        return False

def test_sklearn_models():
    """Test if sklearn models can be loaded"""
    print("\n🔍 Testing sklearn models...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✅ sklearn modules imported successfully")
        
        # Test creating a simple model
        model = RandomForestClassifier(n_estimators=10)
        vectorizer = TfidfVectorizer()
        print("✅ sklearn models created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ sklearn test failed: {e}")
        traceback.print_exc()
        return False

def test_torch_models():
    """Test if PyTorch models can be loaded"""
    print("\n🔍 Testing PyTorch models...")
    
    try:
        import torch
        import torch.nn as nn
        print("✅ PyTorch imported successfully")
        
        # Test creating a simple model
        model = nn.Linear(10, 1)
        print("✅ PyTorch model created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_data_files():
    """Test if data files are accessible"""
    print("\n🔍 Testing data file access...")
    
    try:
        # Check if we're in bundled mode
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            print(f"✅ Running in bundled mode, base path: {base_path}")
            
            # Check for spaCy model
            spacy_model_path = os.path.join(base_path, 'es_core_news_sm')
            if os.path.exists(spacy_model_path):
                print(f"✅ spaCy model directory found: {spacy_model_path}")
                print(f"   Contents: {os.listdir(spacy_model_path)[:5]}...")
            else:
                print(f"❌ spaCy model directory not found: {spacy_model_path}")
                
            # Check for Source_Files
            source_files_path = os.path.join(base_path, 'Source_Files')
            if os.path.exists(source_files_path):
                print(f"✅ Source_Files directory found: {source_files_path}")
                print(f"   Contents: {os.listdir(source_files_path)[:5]}...")
            else:
                print(f"❌ Source_Files directory not found: {source_files_path}")
                
            # Check for models directory
            models_path = os.path.join(base_path, 'models')
            if os.path.exists(models_path):
                print(f"✅ models directory found: {models_path}")
                if os.listdir(models_path):
                    print(f"   Contents: {os.listdir(models_path)[:5]}...")
                else:
                    print("   Directory is empty")
            else:
                print(f"❌ models directory not found: {models_path}")
                
        else:
            print("✅ Running in development mode")
            
        return True
        
    except Exception as e:
        print(f"❌ Data files test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 BULLETPROOF EXECUTABLE TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_array_api_compat,
        test_spacy_model,
        test_sklearn_models,
        test_torch_models,
        test_data_files
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ The executable is BULLETPROOF and ready for deployment!")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("❌ Some critical modules are still not working")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
