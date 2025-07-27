"""
Visualization tools for backtesting results
"""

from .base_visualizer import BacktestVisualizer
from .equity_curve import plot_equity_curve
from .drawdown import plot_drawdown
from .returns import plot_returns_distribution, plot_monthly_returns_heatmap
from .trade_analysis import plot_trade_analysis, plot_trade_duration_histogram
from .rolling_metrics import plot_rolling_metrics

__all__ = [
    'BacktestVisualizer',
    'plot_equity_curve',
    'plot_drawdown',
    'plot_returns_distribution',
    'plot_monthly_returns_heatmap',
    'plot_trade_analysis',
    'plot_trade_duration_histogram',
    'plot_rolling_metrics'
]
