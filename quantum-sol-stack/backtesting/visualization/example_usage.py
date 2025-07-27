"""
Example usage and testing scaffold for QuantumSol Visualization System.

This script demonstrates how to generate each visualization from sample backtest data,
and includes basic test cases to verify functionality and error handling.

Features:
- Generates sample backtest data with realistic metrics
- Demonstrates all visualization modules
- Includes basic test cases for each visualization

Usage:
    python -m backtesting.visualization.example_usage

Next steps:
1. Replace sample data with real backtest results
2. Customize visualizations with your own styles
3. Integrate into your backtesting pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Import visualization modules
from backtesting.visualization.base_visualizer import BacktestVisualizer
from backtesting.visualization.equity_curve import plot_equity_curve
from backtesting.visualization.drawdown import plot_drawdown, plot_drawdown_periods
from backtesting.visualization.returns import plot_returns_distribution
from backtesting.visualization.trade_analysis import plot_trade_performance
from backtesting.visualization.rolling_metrics import plot_rolling_metrics

@dataclass
class BacktestResult:
    """Container for backtest results and metrics"""
    portfolio_history: pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.DataFrame
    drawdown_series: Optional[pd.Series] = None


def generate_sample_data() -> BacktestResult:
    """Generate sample backtest data for visualization examples"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
    
    # Generate equity curve with realistic characteristics
    daily_returns = np.random.normal(0.0005, 0.01, 100)
    equity = 100000 * (1 + np.cumsum(daily_returns))
    
    # Create portfolio history DataFrame
    portfolio_df = pd.DataFrame({
        'date': dates,
        'total_value': equity,
        'cash': np.linspace(100000, 50000, 100),
        'position_value': equity - np.linspace(100000, 50000, 100),
        'returns': daily_returns,
        'benchmark': 100000 * (1 + np.cumsum(np.random.normal(0.0003, 0.008, 100)))
    }).set_index('date')
    
    # Generate sample trades
    trades = pd.DataFrame({
        'entry_time': dates[::5][:15],  # 15 trades
        'exit_time': [d + timedelta(days=np.random.randint(1, 10)) for d in dates[::5][:15]],
        'pnl': np.random.normal(500, 200, 15),
        'duration': np.random.randint(1, 10, 15),
        'entry_price': np.random.uniform(100, 200, 15),
        'exit_price': np.random.uniform(90, 210, 15),
        'symbol': ['AAPL'] * 15,
        'quantity': np.random.randint(10, 100, 15),
        'side': ['long' if x > 0.5 else 'short' for x in np.random.random(15)],
        'win': np.random.random(15) > 0.4
    })
    
    # Generate performance metrics
    metrics = pd.DataFrame({
        'sharpe': np.random.uniform(1.5, 2.5, 100).cumsum() / np.arange(1, 101),
        'sortino': np.random.uniform(1.8, 2.8, 100).cumsum() / np.arange(1, 101),
        'volatility': np.random.uniform(0.01, 0.05, 100),
        'drawdown': np.random.uniform(0, 0.15, 100),
        'win_rate': np.linspace(0.45, 0.65, 100)
    }, index=dates)
    
    # Calculate drawdown series
    roll_max = portfolio_df['total_value'].cummax()
    drawdown_series = (portfolio_df['total_value'] - roll_max) / roll_max
    
    return BacktestResult(
        portfolio_history=portfolio_df,
        trades=trades,
        metrics=metrics,
        drawdown_series=drawdown_series
    )

def example_usage():
    """Demonstrate all visualization components with sample data"""
    # Initialize visualizer with custom style
    visualizer = BacktestVisualizer(style='seaborn-v0_8', template='plotly_white')
    
    # Generate sample data
    result = generate_sample_data()
    
    # 1. Equity Curve
    fig_eq = plot_equity_curve(
        result=result,
        visualizer=visualizer,
        benchmark_col='benchmark',
        show_drawdown=True
    )
    fig_eq.show()
    
    # 2. Drawdown Analysis
    fig_dd = plot_drawdown(
        result=result,
        visualizer=visualizer,
        show_underwater=True,
        show_periods=True
    )
    fig_dd.show()
    
    # 3. Returns Distribution
    fig_ret = plot_returns_distribution(
        result=result,
        visualizer=visualizer,
        bins=30
    )
    fig_ret.show()
    
    # 4. Trade Analysis
    fig_trades = plot_trade_performance(
        result=result,
        visualizer=visualizer,
        pnl_col='pnl',
        win_col='win',
        duration_col='duration'
    )
    fig_trades.show()
    
    # 5. Rolling Metrics
    fig_metrics = plot_rolling_metrics(
        result=result,
        visualizer=visualizer,
        metrics=['sharpe', 'sortino', 'volatility'],
        window=20
    )
    fig_metrics.show()

def run_basic_tests():
    """Run basic tests to verify visualization functions"""
    try:
        visualizer = BacktestVisualizer()
        result = generate_sample_data()
        
        # Test each visualization
        assert plot_equity_curve(result, visualizer) is not None
        assert plot_drawdown(result, visualizer) is not None
        assert plot_returns_distribution(result, visualizer) is not None
        assert plot_trade_performance(result, visualizer) is not None
        assert plot_rolling_metrics(result, visualizer) is not None
        
        print("✅ All visualization tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running QuantumSol visualization examples...")
    
    # Run examples
    print("\n📊 Generating example visualizations...")
    example_usage()
    
    # Run tests
    print("\n🧪 Running tests...")
    run_basic_tests()
    
    print("\n✨ Done! Check the plots that were generated.")
