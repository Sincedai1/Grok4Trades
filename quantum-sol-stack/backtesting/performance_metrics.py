"""
Performance metrics calculation for backtesting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import logging

from .portfolio_simulator import Trade

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for trading strategies
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize PerformanceMetrics
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(
        self, 
        returns: pd.Series, 
        trades: List[Trade],
        initial_capital: float,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Series of portfolio returns
            trades: List of Trade objects
            initial_capital: Initial capital
            benchmark_returns: Optional Series of benchmark returns
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns, initial_capital))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Trade-based metrics
        metrics.update(self._calculate_trade_metrics(trades))
        
        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        # Additional metrics
        metrics.update(self._calculate_additional_metrics(returns, trades))
        
        return metrics
    
    def _calculate_return_metrics(
        self, 
        returns: pd.Series, 
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate return-based metrics"""
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'cumulative_return': 0.0,
                'avg_daily_return': 0.0,
                'median_daily_return': 0.0,
                'best_day': 0.0,
                'worst_day': 0.0
            }
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return
        trading_days = len(returns)
        years = trading_days / 252  # Assuming 252 trading days per year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Cumulative return
        cumulative_return = returns.cumsum().iloc[-1] if len(returns) > 0 else 0
        
        # Daily statistics
        daily_returns = returns.copy()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_return,
            'avg_daily_return': daily_returns.mean(),
            'median_daily_return': daily_returns.median(),
            'best_day': daily_returns.max(),
            'worst_day': daily_returns.min()
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-based metrics"""
        if len(returns) == 0:
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'tail_ratio': 0.0
            }
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(252) 
            if len(downside_returns) > 0 else 0
        )
        sortino_ratio = (
            excess_returns / downside_deviation 
            if downside_deviation > 0 else 0
        )
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio (return to max drawdown)
        calmar_ratio = (
            (annualized_return := self._annualize_returns(returns)) / abs(max_drawdown) 
            if max_drawdown < 0 else 0
        )
        
        # Value at Risk (95%)
        var_95 = returns.quantile(0.05)
        
        # Conditional Value at Risk (95%)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio (absolute value of the 95th percentile / 5th percentile)
        tail_ratio = abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': abs(max_drawdown),
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio
        }
    
    def _calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate trade-based metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'breakeven_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_win_loss_ratio': 0.0,
                'profit_factor': 0.0,
                'avg_trade_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'avg_holding_period': 0.0
            }
        
        # Calculate PnL for each trade
        pnls = []
        for trade in trades:
            pnl = getattr(trade, 'pnl', 0.0)
            pnls.append(pnl)
        
        pnls = np.array(pnls)
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]
        breakeven_trades = pnls[pnls == 0]
        
        # Trade statistics
        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        num_breakeven = len(breakeven_trades)
        
        win_rate = num_winning / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades.mean() if num_winning > 0 else 0
        avg_loss = losing_trades.mean() if num_losing > 0 else 0
        
        # Average win/loss ratio
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Profit factor
        gross_profit = winning_trades.sum() if num_winning > 0 else 0
        gross_loss = abs(losing_trades.sum()) if num_losing > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade PnL
        avg_trade_pnl = pnls.mean()
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_wins_losses(trades)
        
        # Average holding period (if trade has entry and exit times)
        holding_periods = []
        for trade in trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time') and trade.entry_time and trade.exit_time:
                holding_periods.append((trade.exit_time - trade.entry_time).total_seconds() / 3600)  # in hours
        
        avg_holding_period = sum(holding_periods) / len(holding_periods) if holding_periods else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'breakeven_trades': num_breakeven,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade_pnl,
            'largest_win': winning_trades.max() if num_winning > 0 else 0,
            'largest_loss': losing_trades.min() if num_losing > 0 else 0,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'avg_holding_period': avg_holding_period,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _calculate_benchmark_metrics(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return {
                'alpha': 0.0,
                'beta': 0.0,
                'correlation': 0.0,
                'r_squared': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0,
                'up_capture': 0.0,
                'down_capture': 0.0,
                'upside_capture_ratio': 0.0,
                'downside_capture_ratio': 0.0
            }
        
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return {
                'alpha': 0.0,
                'beta': 0.0,
                'correlation': 0.0,
                'r_squared': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0,
                'up_capture': 0.0,
                'down_capture': 0.0,
                'upside_capture_ratio': 0.0,
                'downside_capture_ratio': 0.0
            }
        
        # Beta calculation
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation (CAPM)
        portfolio_return = aligned_returns.mean() * 252
        benchmark_return = aligned_benchmark.mean() * 252
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Correlation and R-squared
        correlation = aligned_returns.corr(aligned_benchmark)
        r_squared = correlation ** 2
        
        # Tracking error
        active_returns = aligned_returns - aligned_benchmark
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = (
            (aligned_returns - aligned_benchmark).mean() * np.sqrt(252) / 
            (aligned_returns - aligned_benchmark).std()
        ) if (aligned_returns - aligned_benchmark).std() > 0 else 0
        
        # Up/Down capture ratios
        up_market = aligned_benchmark > 0
        down_market = aligned_benchmark < 0
        
        up_capture = (
            (1 + aligned_returns[up_market]).prod() - 1
        ) if any(up_market) else 0
        
        down_capture = (
            (1 + aligned_returns[down_market]).prod() - 1
        ) if any(down_market) else 0
        
        benchmark_up = (
            (1 + aligned_benchmark[up_market]).prod() - 1
        ) if any(up_market) else 0
        
        benchmark_down = (
            (1 + aligned_benchmark[down_market]).prod() - 1
        ) if any(down_market) else 0
        
        upside_capture_ratio = up_capture / benchmark_up if benchmark_up != 0 else 0
        downside_capture_ratio = down_capture / benchmark_down if benchmark_down != 0 else 0
        
        return {
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'upside_capture_ratio': upside_capture_ratio,
            'downside_capture_ratio': downside_capture_ratio
        }
    
    def _calculate_additional_metrics(
        self, 
        returns: pd.Series, 
        trades: List[Trade]
    ) -> Dict[str, float]:
        """Calculate additional performance metrics"""
        if len(returns) == 0:
            return {
                'k_ratio': 0.0,
                'omega_ratio': 0.0,
                'gain_to_pain_ratio': 0.0,
                'common_sense_ratio': 0.0,
                'tail_ratio': 0.0,
                'daily_value_at_risk': 0.0,
                'conditional_value_at_risk': 0.0,
                'ulcer_index': 0.0,
                'serenity_ratio': 0.0,
                'burke_ratio': 0.0,
                'pain_index': 0.0,
                'pain_ratio': 0.0
            }
        
        # K-Ratio (slope of equity curve)
        k_ratio = self._calculate_k_ratio(returns)
        
        # Omega Ratio
        omega_ratio = self._calculate_omega_ratio(returns)
        
        # Gain to Pain Ratio
        gain_to_pain_ratio = self._calculate_gain_to_pain_ratio(returns)
        
        # Common Sense Ratio (CSR)
        common_sense_ratio = self._calculate_common_sense_ratio(returns, trades)
        
        # Tail Ratio
        tail_ratio = self._calculate_tail_ratio(returns)
        
        # Daily Value at Risk (95%)
        daily_var = returns.quantile(0.05)
        
        # Conditional Value at Risk (95%)
        cvar = returns[returns <= daily_var].mean() if len(returns[returns <= daily_var]) > 0 else 0
        
        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(returns)
        
        # Serenity Index
        serenity_ratio = self._calculate_serenity_ratio(returns)
        
        # Burke Ratio
        burke_ratio = self._calculate_burke_ratio(returns)
        
        # Pain Index and Pain Ratio
        pain_index, pain_ratio = self._calculate_pain_metrics(returns)
        
        return {
            'k_ratio': k_ratio,
            'omega_ratio': omega_ratio,
            'gain_to_pain_ratio': gain_to_pain_ratio,
            'common_sense_ratio': common_sense_ratio,
            'tail_ratio': tail_ratio,
            'daily_value_at_risk': daily_var,
            'conditional_value_at_risk': cvar,
            'ulcer_index': ulcer_index,
            'serenity_ratio': serenity_ratio,
            'burke_ratio': burke_ratio,
            'pain_index': pain_index,
            'pain_ratio': pain_ratio
        }
    
    def _calculate_consecutive_wins_losses(
        self, 
        trades: List[Trade]
    ) -> Tuple[int, int]:
        """Calculate consecutive wins and losses"""
        if not trades:
            return 0, 0
        
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            pnl = getattr(trade, 'pnl', 0.0)
            
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return max_consec_wins, max_consec_losses
    
    def _annualize_returns(self, returns: pd.Series) -> float:
        """Annualize returns"""
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252  # 252 trading days per year
        
        if years <= 0:
            return 0.0
            
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_k_ratio(self, returns: pd.Series) -> float:
        """Calculate K-Ratio (slope of equity curve)"""
        if len(returns) < 2:
            return 0.0
        
        n = len(returns)
        x = np.arange(n)
        y = (1 + returns).cumprod()
        
        # Linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        # Scale to annual basis
        k_ratio = slope * np.sqrt(252) / (y.std() * np.sqrt(n))
        
        return k_ratio
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - threshold
        upside = excess_returns[excess_returns > 0].sum()
        downside = abs(excess_returns[excess_returns < 0].sum())
        
        return upside / downside if downside > 0 else float('inf')
    
    def _calculate_gain_to_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate Gain to Pain Ratio"""
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        sum_negative_returns = returns[returns < 0].sum()
        
        return total_return / abs(sum_negative_returns) if sum_negative_returns < 0 else float('inf')
    
    def _calculate_common_sense_ratio(
        self, 
        returns: pd.Series, 
        trades: List[Trade]
    ) -> float:
        """Calculate Common Sense Ratio"""
        if len(returns) == 0 or not trades:
            return 0.0
        
        # Calculate percentage of positive returns
        percent_positive = (returns > 0).mean()
        
        # Calculate average win/loss ratio from trades
        pnls = [getattr(t, 'pnl', 0.0) for t in trades]
        if not pnls:
            return 0.0
            
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = abs(np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0
        
        if avg_loss == 0:
            return 0.0
            
        win_loss_ratio = avg_win / avg_loss
        
        # Common Sense Ratio = % Positive Returns * Win/Loss Ratio
        return percent_positive * win_loss_ratio
    
    def _calculate_tail_ratio(self, returns: pd.Series, quantile: float = 0.05) -> float:
        """Calculate Tail Ratio"""
        if len(returns) == 0:
            return 0.0
            
        right_tail = returns.quantile(1 - quantile)
        left_tail = abs(returns.quantile(quantile))
        
        return right_tail / left_tail if left_tail > 0 else 0.0
    
    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index"""
        if len(returns) == 0:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        return np.sqrt((drawdowns ** 2).mean())
    
    def _calculate_serenity_ratio(self, returns: pd.Series) -> float:
        """Calculate Serenity Index (Return / (Volatility * Max Drawdown))"""
        if len(returns) == 0:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdowns.min())
        
        if max_drawdown == 0:
            return 0.0
            
        annual_return = self._annualize_returns(returns)
        volatility = returns.std() * np.sqrt(252)
        
        return annual_return / (volatility * max_drawdown) if volatility > 0 else 0.0
    
    def _calculate_burke_ratio(self, returns: pd.Series) -> float:
        """Calculate Burke Ratio (Return / sqrt(sum of squared drawdowns))"""
        if len(returns) == 0:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        sum_squared_drawdowns = (drawdowns ** 2).sum()
        if sum_squared_drawdowns <= 0:
            return 0.0
            
        annual_return = self._annualize_returns(returns)
        
        return annual_return / np.sqrt(sum_squared_drawdowns)
    
    def _calculate_pain_metrics(
        self, 
        returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate Pain Index and Pain Ratio"""
        if len(returns) == 0:
            return 0.0, 0.0
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (running_max - cumulative) / running_max
        
        # Pain Index (average percentage drawdown)
        pain_index = drawdowns.mean()
        
        # Pain Ratio (return / pain index)
        annual_return = self._annualize_returns(returns)
        pain_ratio = annual_return / pain_index if pain_index > 0 else 0.0
        
        return pain_index, pain_ratio
    
    def calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve"""
        if len(equity_curve) == 0:
            return pd.Series(dtype=float)
            
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def calculate_rolling_metrics(
        self, 
        returns: pd.Series, 
        window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            returns: Series of returns
            window: Rolling window size in periods
            
        Returns:
            DataFrame with rolling metrics
        """
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window).sum()
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        rolling_excess = returns.rolling(window).mean() * 252 - self.risk_free_rate
        rolling_metrics['rolling_sharpe'] = rolling_excess / rolling_metrics['rolling_volatility']
        
        # Rolling max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_drawdown = (cumulative - rolling_max) / rolling_max
        rolling_metrics['rolling_max_drawdown'] = rolling_drawdown.rolling(window).min()
        
        # Rolling Sortino ratio
        rolling_downside = returns.rolling(window).apply(
            lambda x: x[x < 0].std() * np.sqrt(252) if len(x[x < 0]) > 0 else 0
        )
        rolling_metrics['rolling_sortino'] = rolling_excess / rolling_downside
        
        return rolling_metrics.dropna()
