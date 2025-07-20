#!/usr/bin/env python3
"""
Test script to verify the gender agreement fixes for Spanish automotive terms.

This script tests the specific cases reported by the user:
- "GUIA LATERAL IZQUIERDA PARAGOLPES DELANTERA" → should be "DELANTERO"
- "BROCHES GUARDAPOLVO PLASTICO DELANTERO DERECHA" → should be "DERECHO"
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.text_utils import normalize_text, expand_linguistic_variations_text

def test_gender_agreement_fixes():
    """Test the specific gender agreement issues reported by the user"""
    
    print("🔧 Testing Gender Agreement Fixes")
    print("=" * 60)
    
    # Test cases from user feedback
    test_cases = [
        {
            "input": "GUIA LATERAL IZQUIERDA PARAGOLPES DELANTERA",
            "expected_fix": "GUIA LATERAL IZQUIERDA PARAGOLPES DELANTERO",
            "explanation": "DELANTERO should agree with PARAGOLPES (masculine), not GUIA (feminine)"
        },
        {
            "input": "BROCHES GUARDAPOLVO PLASTICO DELANTERO DERECHA", 
            "expected_fix": "BROCHES GUARDAPOLVO PLASTICO DELANTERO DERECHO",
            "explanation": "DERECHO should agree with GUARDAPOLVO (masculine), not BROCHES (masculine plural)"
        },
        # Additional test cases to verify the fix
        {
            "input": "FAROLA IZQUIERDA PARAGOLPES DELANTERA",
            "expected_fix": "FAROLA IZQUIERDA PARAGOLPES DELANTERO", 
            "explanation": "DELANTERO should agree with PARAGOLPES (masculine)"
        },
        {
            "input": "ESPEJO LATERAL DERECHO GUIA IZQUIERDA",
            "expected_fix": "ESPEJO LATERAL DERECHO GUIA IZQUIERDA",
            "explanation": "Each adjective agrees with its immediate noun"
        },
        {
            "input": "GUARDAPOLVO PLASTICO DELANTERO IZQUIERDA",
            "expected_fix": "GUARDAPOLVO PLASTICO DELANTERO IZQUIERDO",
            "explanation": "IZQUIERDO should agree with GUARDAPOLVO (masculine)"
        }
    ]
    
    print("\n📋 Test Results:")
    print("-" * 60)
    
    correct_fixes = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case["input"]
        expected = test_case["expected_fix"]
        explanation = test_case["explanation"]
        
        print(f"\n🧪 Test {i}: Gender Agreement Fix")
        print(f"   Input: '{input_text}'")
        print(f"   Expected: '{expected}'")
        print(f"   Explanation: {explanation}")
        
        # Apply linguistic expansion (this includes gender agreement fixes)
        result = expand_linguistic_variations_text(input_text.lower())
        
        print(f"   Result: '{result}'")
        
        # Normalize both for comparison (case-insensitive)
        result_normalized = result.lower().strip()
        expected_normalized = expected.lower().strip()
        
        if result_normalized == expected_normalized:
            print(f"   ✅ FIXED: Gender agreement is now correct!")
            correct_fixes += 1
        else:
            print(f"   ❌ STILL WRONG: Gender agreement needs more work")
            
            # Show the difference
            result_words = result_normalized.split()
            expected_words = expected_normalized.split()
            
            print(f"   📊 Word-by-word comparison:")
            max_len = max(len(result_words), len(expected_words))
            for j in range(max_len):
                result_word = result_words[j] if j < len(result_words) else "---"
                expected_word = expected_words[j] if j < len(expected_words) else "---"
                
                if result_word == expected_word:
                    print(f"      {j+1}: '{result_word}' ✅")
                else:
                    print(f"      {j+1}: '{result_word}' vs '{expected_word}' ❌")
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY: {correct_fixes}/{total_tests} gender agreement issues fixed")
    
    success_rate = (correct_fixes / total_tests) * 100
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("✅ EXCELLENT: Gender agreement fixes are working!")
    elif success_rate >= 60:
        print("⚠️  GOOD: Most issues fixed, some edge cases remain")
    else:
        print("❌ NEEDS MORE WORK: Gender agreement logic still has issues")
    
    return success_rate

def test_individual_components():
    """Test individual components of the gender agreement system"""
    
    print("\n\n🔬 Testing Individual Gender Agreement Components")
    print("=" * 60)
    
    from utils.text_utils import get_noun_gender, find_immediate_noun_for_adjective
    
    # Test noun gender detection
    print("\n📋 Noun Gender Detection:")
    print("-" * 30)
    
    gender_tests = [
        ("guia", "feminine"),
        ("paragolpes", "masculine"), 
        ("guardapolvo", "masculine"),
        ("broches", "masculine"),
        ("farola", "feminine"),
        ("espejo", "masculine")
    ]
    
    for noun, expected_gender in gender_tests:
        detected_gender = get_noun_gender(noun)
        status = "✅" if detected_gender == expected_gender else "❌"
        print(f"   {noun}: {detected_gender} (expected: {expected_gender}) {status}")
    
    # Test immediate noun finding
    print("\n📋 Immediate Noun Finding:")
    print("-" * 30)
    
    noun_finding_tests = [
        (["guia", "lateral", "izquierda", "paragolpes", "delantera"], 4, "paragolpes"),
        (["broches", "guardapolvo", "plastico", "delantero", "derecha"], 4, "guardapolvo"),
        (["farola", "izquierda", "paragolpes", "delantero"], 3, "paragolpes"),
    ]
    
    for words, index, expected_noun in noun_finding_tests:
        found_noun = find_immediate_noun_for_adjective(words, index)
        status = "✅" if found_noun == expected_noun else "❌"
        print(f"   Words: {words}")
        print(f"   Index {index} ('{words[index]}'): found '{found_noun}' (expected: '{expected_noun}') {status}")

if __name__ == "__main__":
    print("🚀 Starting Gender Agreement Fix Tests")
    print("=" * 80)
    
    try:
        # Run main tests
        success_rate = test_gender_agreement_fixes()
        
        # Run component tests
        test_individual_components()
        
        print("\n" + "=" * 80)
        print("🏁 Testing Complete!")
        
        if success_rate >= 80:
            print("🎉 The gender agreement fixes are working excellently!")
            print("   Spanish automotive terms should now have correct gender agreement.")
        else:
            print("⚠️  The gender agreement system needs further refinement.")
            print("   Some cases are still not handled correctly.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
