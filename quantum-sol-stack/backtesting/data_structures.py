"""
Data structures for backtesting results and configurations.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd

@dataclass
class BacktestResult:
    """
    Container for backtest results and metrics.
    
    Attributes:
        portfolio_history: DataFrame containing the portfolio value over time
        trades: DataFrame containing all executed trades
        metrics: DataFrame containing performance metrics over time
        drawdown_series: Optional Series containing drawdown values
    """
    portfolio_history: pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.DataFrame
    drawdown_series: Optional[pd.Series] = None
