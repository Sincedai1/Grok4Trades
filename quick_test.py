#!/usr/bin/env python3
"""
Quick Test for Grok4Trades Optimizations

This script provides a lightweight way to verify the core functionality
of the optimized trading bot components without requiring all dependencies.
"""

import time
import sys
from pathlib import Path

def print_test_result(name: str, success: bool, details: str = ""):
    """Helper to print test results"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {name}" + (f" - {details}" if details else ""))

def test_imports() -> bool:
    """Test if core modules can be imported"""
    print("\n=== Testing Core Imports ===")
    tests = {
        "numpy": False,
        "core.minimal_engine": False,
        "strategies.simple_ma": False
    }
    
    # Test numpy
    try:
        import numpy as np
        tests["numpy"] = True
        print_test_result("numpy", True, f"v{np.__version__}")
    except ImportError:
        print_test_result("numpy", False, "Not installed")
    
    # Test core modules
    try:
        from core.minimal_engine import MinimalTradingBot
        tests["core.minimal_engine"] = True
        print_test_result("core.minimal_engine", True)
    except Exception as e:
        print_test_result("core.minimal_engine", False, str(e))
    
    try:
        from strategies.simple_ma import SimpleMAStrategy
        tests["strategies.simple_ma"] = True
        print_test_result("strategies.simple_ma", True)
    except Exception as e:
        print_test_result("strategies.simple_ma", False, str(e))
    
    return all(tests.values())

def test_simple_ma_performance():
    """Test SimpleMAStrategy performance"""
    print("\n=== Testing SimpleMAStrategy Performance ===")
    try:
        from strategies.simple_ma import SimpleMAStrategy
        import numpy as np
        
        # Create test data
        prices = np.random.normal(50000, 1000, 1000)
        strategy = SimpleMAStrategy(fast_window=10, slow_window=20)
        
        # Warm up
        for _ in range(10):
            strategy.generate_signal({'close': prices[:100]})
        
        # Benchmark
        start_time = time.time()
        iterations = 1000
        for _ in range(iterations):
            strategy.generate_signal({'close': np.random.normal(50000, 1000, 100)})
        
        elapsed = (time.time() - start_time) * 1000  # ms
        avg_time = elapsed / iterations
        
        print_test_result("Signal Generation", avg_time < 5, 
                         f"{avg_time:.4f}ms per signal")
        return avg_time < 5
        
    except Exception as e:
        print_test_result("Signal Generation", False, str(e))
        return False

def main():
    """Run all tests"""
    print("\n=== Grok4Trades Optimization Quick Test ===")
    
    # Run import tests
    imports_ok = test_imports()
    
    # Only run performance tests if imports succeeded
    perf_ok = False
    if imports_ok:
        perf_ok = test_simple_ma_performance()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Core Imports: {'✅' if imports_ok else '❌'}")
    print(f"Performance:   {'✅' if perf_ok else '❌'}")
    
    if imports_ok and perf_ok:
        print("\n✅ All tests passed! The optimizations appear to be working correctly.")
        print("You can now run the full benchmark with: python benchmark_performance.py")
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
        print("\nNEXT STEPS:")
        print("1. Install required packages: pip install numpy loguru")
        print("2. Check Python environment and dependencies")
        print("3. Run tests again")

if __name__ == "__main__":
    main()
