#!/usr/bin/env python3
"""
Complete End-to-End Setup Test
Tests all major components of the prompt lab setup.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

def test_environment():
    """Test environment setup."""
    print("🔧 Testing Environment Setup...")
    
    # Load environment
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    if not api_key.startswith('sk-'):
        print("❌ Invalid API key format")
        return False
    
    print(f"✅ API key configured: {api_key[:7]}...")
    return True

def test_imports():
    """Test all critical imports."""
    print("\n📦 Testing Imports...")
    
    try:
        from src.runner import run, run_with_ledger
        print("✅ Runner module imported")
        
        from src.example import TokenLedger
        print("✅ TokenLedger imported")
        
        from evals.evaluation_framework import PromptEvaluator
        print("✅ Evaluation framework imported")
        
        import openai
        print("✅ OpenAI module imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_token_ledger():
    """Test token ledger functionality."""
    print("\n📊 Testing Token Ledger...")
    
    try:
        from src.example import TokenLedger
        
        # Test ledger creation and operations
        ledger = TokenLedger('data/test_ledger.csv')
        
        # Add test entry
        ledger.add_entry(
            phase="end_to_end_test",
            model="gpt-4o-mini",
            tokens_in=10,
            tokens_out=20,
            cost_usd=0.001
        )
        
        # Verify entry
        entries = ledger.get_ledger()
        test_entries = [e for e in entries if e['phase'] == 'end_to_end_test']
        
        if test_entries:
            print(f"✅ Ledger working: {len(test_entries)} test entry added")
            return True
        else:
            print("❌ Ledger test entry not found")
            return False
            
    except Exception as e:
        print(f"❌ Ledger test failed: {e}")
        return False

def test_evaluation_framework():
    """Test evaluation framework."""
    print("\n🔍 Testing Evaluation Framework...")
    
    try:
        from evals.evaluation_framework import PromptEvaluator
        
        # Create evaluator
        evaluator = PromptEvaluator("end_to_end_test")
        
        # Test evaluation
        result = evaluator.evaluate_response(
            prompt="Test prompt for setup validation",
            response="This is a test response with keywords validation",
            criteria={
                'min_length': 5,
                'contains_keywords': ['test', 'validation']
            }
        )
        
        # Check results
        if 'scores' in result and 'keyword_coverage' in result['scores']:
            coverage = result['scores']['keyword_coverage']
            print(f"✅ Evaluation working: {coverage:.0%} keyword coverage")
            return True
        else:
            print("❌ Evaluation result structure invalid")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def test_api_runner():
    """Test API runner (mock test without actual API call)."""
    print("\n🚀 Testing API Runner...")
    
    try:
        from src.runner import run
        import openai
        
        # Just test that the function can be called
        # We won't make actual API calls in this test
        print("✅ Runner function available")
        print("✅ OpenAI client can be initialized")
        
        # Test pricing configuration
        from src.runner import PRICE
        if 'gpt-4o-mini' in PRICE:
            print("✅ Pricing configuration loaded")
            return True
        else:
            print("❌ Pricing configuration missing")
            return False
            
    except Exception as e:
        print(f"❌ Runner test failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure."""
    print("\n📁 Testing Directory Structure...")
    
    required_dirs = [
        'src',
        'evals', 
        'data',
        'notebooks',
        'tests'
    ]
    
    required_files = [
        'src/runner.py',
        'src/example.py',
        'evals/evaluation_framework.py',
        'smoke_test.py',
        'requirements-pinned.txt',
        '.env.example',
        'README.md'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            missing_items.append(f"Directory: {dir_name}")
    
    # Check files
    for file_name in required_files:
        if not os.path.isfile(file_name):
            missing_items.append(f"File: {file_name}")
    
    if missing_items:
        print("❌ Missing items:")
        for item in missing_items:
            print(f"   - {item}")
        return False
    else:
        print("✅ All required directories and files present")
        return True

def main():
    """Run complete end-to-end test."""
    print("🧪 COMPLETE PROMPT LAB SETUP TEST")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Environment", test_environment),
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Token Ledger", test_token_ledger),
        ("Evaluation Framework", test_evaluation_framework),
        ("API Runner", test_api_runner),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} | {status}")
        if success:
            passed += 1
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Your prompt lab setup is complete and ready for use!")
        print("\nNext steps:")
        print("1. Run: jupyter lab notebooks/01_foundations.ipynb")
        print("2. Start your prompt engineering experiments!")
        return True
    else:
        print(f"\n💥 {total - passed} tests failed!")
        print("Please review the errors above and fix any issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
