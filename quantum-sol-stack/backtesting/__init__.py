"""
QuantumSol Backtesting Framework
Comprehensive backtesting system for AI trading agents
"""

from .backtest_engine import BacktestEngine
from .performance_metrics import PerformanceMetrics
from .portfolio_simulator import PortfolioSimulator, Position
from .visualization import BacktestVisualizer
from .data_structures import BacktestResult

__version__ = "1.0.0"
__all__ = [
    "BacktestEngine",
    "PerformanceMetrics",
    "PortfolioSimulator",
    "Position",
    "BacktestVisualizer",
    "BacktestResult"
]
