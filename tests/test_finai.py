#!/usr/bin/env python3
"""
Quick test script for FinAI
Tests basic functionality without full training
"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    try:
        import numpy
        print("✓ numpy")
        import pandas
        print("✓ pandas")
        import sklearn
        print("✓ scikit-learn")
        try:
            import nltk
            print("✓ nltk")
        except ImportError:
            print("⚠ nltk not available (optional)")
        try:
            import numpy_financial
            print("✓ numpy-financial")
        except ImportError:
            print("⚠ numpy-financial not available (optional)")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False

def test_finai_structure():
    """Test that FinAI script has correct structure"""
    print("\nTesting FinAI structure...")
    try:
        # Import without running
        import finai
        
        # Check main classes exist
        assert hasattr(finai, 'FinAI'), "FinAI class not found"
        print("✓ FinAI class")
        
        assert hasattr(finai, 'FinAIModel'), "FinAIModel class not found"
        print("✓ FinAIModel class")
        
        assert hasattr(finai, 'SyntheticDataGenerator'), "SyntheticDataGenerator not found"
        print("✓ SyntheticDataGenerator class")
        
        assert hasattr(finai, 'FinancialCalculators'), "FinancialCalculators not found"
        print("✓ FinancialCalculators class")
        
        assert hasattr(finai, 'ResponseGenerator'), "ResponseGenerator not found"
        print("✓ ResponseGenerator class")
        
        return True
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation"""
    print("\nTesting synthetic data generation...")
    try:
        from finai import SyntheticDataGenerator
        
        # Generate small sample
        queries, labels = SyntheticDataGenerator.generate_training_data(100)
        assert len(queries) > 0, "No queries generated"
        assert len(labels) > 0, "No labels generated"
        assert len(queries) == len(labels), "Mismatched query/label counts"
        print(f"✓ Generated {len(queries)} training samples")
        
        # Test financial scenarios
        scenarios = SyntheticDataGenerator.generate_financial_scenarios(10)
        assert len(scenarios) > 0, "No scenarios generated"
        print(f"✓ Generated {len(scenarios)} financial scenarios")
        
        return True
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False

def test_calculators():
    """Test financial calculators"""
    print("\nTesting financial calculators...")
    try:
        from finai import FinancialCalculators
        
        # Test compound interest
        result = FinancialCalculators.compound_interest(10000, 0.07, 10)
        assert result > 10000, "Compound interest calculation failed"
        print(f"✓ Compound interest: $10,000 → ${result:,.2f} in 10 years")
        
        # Test loan payment
        payment = FinancialCalculators.loan_payment(300000, 0.065, 30)
        assert payment > 0, "Loan payment calculation failed"
        print(f"✓ Loan payment: ${payment:,.2f}/month for $300k mortgage")
        
        # Test Monte Carlo (small sample)
        mc_result = FinancialCalculators.monte_carlo_simulation(
            10000, 500, 10, 0.07, 0.15, iterations=100
        )
        assert 'median' in mc_result, "Monte Carlo failed"
        print(f"✓ Monte Carlo simulation: Median ${mc_result['median']:,.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Calculator test failed: {e}")
        return False

def test_nlp():
    """Test NLP processor"""
    print("\nTesting NLP processor...")
    try:
        from finai import NLPProcessor
        
        nlp = NLPProcessor()
        
        # Test tokenization
        tokens = nlp.tokenize("I need help with budgeting")
        assert len(tokens) > 0, "Tokenization failed"
        print(f"✓ Tokenization: {len(tokens)} tokens")
        
        # Test number extraction
        numbers = nlp.extract_numbers("I make $5000 per month and have $10,000 saved")
        assert len(numbers) == 2, "Number extraction failed"
        print(f"✓ Number extraction: Found {numbers}")
        
        # Test sentiment
        sentiment = nlp.detect_sentiment("I'm worried about my debt")
        print(f"✓ Sentiment detection: '{sentiment}'")
        
        return True
    except Exception as e:
        print(f"✗ NLP test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("FinAI Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("FinAI Structure", test_finai_structure),
        ("Synthetic Data", test_synthetic_data),
        ("Financial Calculators", test_calculators),
        ("NLP Processing", test_nlp),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! FinAI is ready to use.")
        print("Run: python finai.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
        print("Install missing packages: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
