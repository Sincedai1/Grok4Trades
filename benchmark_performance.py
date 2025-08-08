#!/usr/bin/env python3
"""
Performance Benchmark for Grok4Trades

This script benchmarks the critical performance metrics of the trading bot
including signal generation speed, memory usage, and startup time.
"""

import time
import asyncio
import tracemalloc
import numpy as np
from loguru import logger
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = str(Path(__file__).parent.absolute())
sys.path.insert(0, PROJECT_ROOT)

# Import components after path setup
from core.minimal_engine import MinimalTradingBot, TradeSignal
from strategies.simple_ma import SimpleMAStrategy
from risk.basic_limits import BasicRiskManager

class MockExchange:
    """Mock exchange for benchmarking"""
    def __init__(self):
        self.prices = np.linspace(45000, 55000, 1000)  # 1000 price points
        self.idx = 0
        
    async def fetch_ticker(self, symbol: str) -> dict:
        """Return a mock ticker with sequential prices"""
        self.idx = (self.idx + 1) % len(self.prices)
        price = self.prices[self.idx]
        return {
            'symbol': symbol,
            'last': price,
            'bid': price * 0.9995,
            'ask': price * 1.0005,
            'timestamp': time.time(),
        }

class Benchmark:
    """Performance benchmarking for trading bot components"""
    
    def __init__(self):
        self.exchange = MockExchange()
        self.strategy = SimpleMAStrategy(fast_window=10, slow_window=20)
        self.bot = MinimalTradingBot(self.exchange, 'BTC/USDT')
        
    async def benchmark_signal_generation(self, iterations: int = 1000) -> float:
        """Benchmark signal generation speed"""
        # Generate test data
        test_data = {
            'close': np.random.normal(50000, 1000, 100)
        }
        
        # Warm-up
        for _ in range(10):
            self.strategy.generate_signal(test_data)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            self.strategy.generate_signal(test_data)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        return elapsed / iterations  # ms per signal
    
    async def benchmark_memory_usage(self) -> float:
        """Benchmark memory usage"""
        tracemalloc.start()
        
        # Create and run bot for a short time
        bot = MinimalTradingBot(self.exchange, 'BTC/USDT')
        
        # Take memory snapshot after initialization
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return peak / (1024 * 1024)  # Convert to MB
    
    async def benchmark_startup_time(self) -> float:
        """Benchmark bot startup time"""
        start = time.perf_counter()
        bot = MinimalTradingBot(self.exchange, 'BTC/USDT')
        elapsed = (time.perf_counter() - start) * 1000  # ms
        return elapsed
    
    async def run_benchmarks(self):
        """Run all benchmarks and report results"""
        logger.info("Starting performance benchmarks...")
        
        # Signal Generation
        signal_time = await self.benchmark_signal_generation(1000)
        logger.info(f"Signal Generation: {signal_time:.4f}ms per signal")
        
        # Memory Usage
        memory_usage = await self.benchmark_memory_usage()
        logger.info(f"Memory Usage: {memory_usage:.2f}MB")
        
        # Startup Time
        startup_time = await self.benchmark_startup_time()
        logger.info(f"Startup Time: {startup_time:.2f}ms")
        
        # Print final results
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*50)
        print(f"Signal Generation: {signal_time:.4f}ms (Target: <5ms) - {"PASS" if signal_time < 5 else "FAIL"}")
        print(f"Memory Usage: {memory_usage:.2f}MB (Target: <100MB) - {"PASS" if memory_usage < 100 else "FAIL"}")
        print(f"Startup Time: {startup_time:.2f}ms (Target: <3000ms) - {"PASS" if startup_time < 3000 else "FAIL"}")
        print("="*50 + "\n")
        
        return {
            'signal_generation_ms': signal_time,
            'memory_usage_mb': memory_usage,
            'startup_time_ms': startup_time
        }

if __name__ == "__main__":
    benchmark = Benchmark()
    asyncio.run(benchmark.run_benchmarks())
