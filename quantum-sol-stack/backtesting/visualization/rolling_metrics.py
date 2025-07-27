"""
Rolling metrics visualization for backtesting results
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data_structures import BacktestResult
from .base_visualizer import BacktestVisualizer

def plot_rolling_metrics(
    result: BacktestResult, 
    visualizer: BacktestVisualizer,
    window: int = 21,
    metrics: List[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plot rolling performance metrics
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        window: Rolling window size in periods
        metrics: List of metrics to plot. Options: 
            - 'sharpe': Rolling Sharpe ratio
            - 'sortino': Rolling Sortino ratio
            - 'volatility': Rolling volatility (annualized)
            - 'drawdown': Rolling maximum drawdown
            - 'win_rate': Rolling win rate (%)
            - 'profit_factor': Rolling profit factor
            - 'avg_trade': Rolling average trade PnL
            
    Returns:
        Plotly Figure object
    """
    if metrics is None:
        metrics = ['sharpe', 'sortino', 'volatility', 'drawdown']
    
    # Get returns data
    portfolio_df = result.portfolio_history.copy()
    
    # Calculate returns if not already present
    if 'returns' not in portfolio_df.columns:
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change().fillna(0)
    
    # Calculate rolling metrics
    rolling_metrics = {}
    
    if 'sharpe' in metrics:
        rolling_metrics['sharpe'] = (
            portfolio_df['returns'].rolling(window).mean() / 
            portfolio_df['returns'].rolling(window).std() * np.sqrt(252)
        )
    
    if 'sortino' in metrics:
        def sortino_ratio(returns, window=window):
            mean = returns.rolling(window).mean()
            std_neg = returns[returns < 0].rolling(window).std() * np.sqrt(252)
            return mean / std_neg
        
        rolling_metrics['sortino'] = portfolio_df['returns'].rolling(window).apply(
            sortino_ratio, raw=False
        )
    
    if 'volatility' in metrics:
        rolling_metrics['volatility'] = (
            portfolio_df['returns'].rolling(window).std() * np.sqrt(252)
        )
    
    if 'drawdown' in metrics:
        rolling_max = portfolio_df['total_value'].rolling(window, min_periods=1).max()
        rolling_dd = (portfolio_df['total_value'] - rolling_max) / rolling_max
        rolling_metrics['drawdown'] = rolling_dd.rolling(window).min()
    
    # Calculate trade-based metrics if trades are available
    if hasattr(result, 'trades') and result.trades:
        trades_df = pd.DataFrame([{
            'exit_time': t.exit_time,
            'pnl': t.pnl if t.pnl is not None else 0,
            'is_win': t.pnl > 0 if t.pnl is not None else False
        } for t in result.trades if t.exit_time is not None])
        
        if not trades_df.empty:
            trades_df.set_index('exit_time', inplace=True)
            
            if 'win_rate' in metrics:
                # Resample to daily and calculate rolling win rate
                daily_trades = trades_df.resample('D').agg({
                    'is_win': ['count', 'sum']
                })
                daily_trades.columns = ['trades', 'wins']
                daily_trades['win_rate'] = daily_trades['wins'] / daily_trades['trades']
                rolling_metrics['win_rate'] = daily_trades['win_rate'].rolling(
                    f'{window}D', min_periods=1
                ).mean()
            
            if 'profit_factor' in metrics:
                # Calculate daily profit factor
                daily_pnl = trades_df['pnl'].resample('D').sum()
                daily_wins = daily_pnl[daily_pnl > 0]
                daily_losses = daily_pnl[daily_pnl < 0].abs()
                
                profit_factor = pd.Series(index=daily_pnl.index, dtype=float)
                profit_factor[daily_wins.index] = daily_wins
                profit_factor[daily_losses.index] = -daily_losses
                
                rolling_metrics['profit_factor'] = (
                    profit_factor.rolling(f'{window}D', min_periods=1).sum() / 
                    daily_losses.rolling(f'{window}D', min_periods=1).sum()
                )
            
            if 'avg_trade' in metrics:
                rolling_metrics['avg_trade'] = (
                    trades_df['pnl'].rolling(f'{window}D', min_periods=1).mean()
                )
    
    # Create figure with subplots
    n_metrics = len(rolling_metrics)
    if n_metrics == 0:
        return _create_empty_plot('No metrics to display')
    
    fig = make_subplots(
        rows=n_metrics, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[_get_metric_name(m) for m in rolling_metrics.keys()]
    )
    
    # Add traces for each metric
    for i, (metric_name, metric_series) in enumerate(rolling_metrics.items(), 1):
        if metric_series is None or metric_series.empty:
            continue
            
        color = _get_metric_color(metric_name, visualizer)
        
        fig.add_trace(
            go.Scatter(
                x=metric_series.index,
                y=metric_series,
                mode='lines',
                name=_get_metric_name(metric_name),
                line=dict(color=color, width=1.5),
                hovertemplate='%{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=i,
            col=1
        )
        
        # Add zero line for ratio metrics
        if metric_name in ['sharpe', 'sortino', 'profit_factor']:
            fig.add_hline(
                y=1 if metric_name == 'profit_factor' else 0,
                line_dash='dash',
                line_color='gray',
                opacity=0.5,
                row=i,
                col=1
            )
        
        # Add mean line
        if not metric_series.isna().all():
            mean_val = metric_series.mean()
            fig.add_hline(
                y=mean_val,
                line_dash='dot',
                line_color=color,
                opacity=0.7,
                annotation_text=f'Mean: {mean_val:.2f}',
                annotation_position='bottom right',
                row=i,
                col=1
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Rolling Metrics (Window: {window} Periods)',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template=visualizer.template,
        hovermode='x',
        plot_bgcolor='white',
        margin=dict(t=80, b=50, l=50, r=50),
        height=250 * n_metrics,
        showlegend=False
    )
    
    # Update y-axes titles
    for i, metric_name in enumerate(rolling_metrics.keys(), 1):
        fig.update_yaxes(
            title_text=_get_metric_units(metric_name),
            row=i,
            col=1
        )
    
    # Update x-axis title for the bottom subplot
    fig.update_xaxes(title_text='Date', row=n_metrics, col=1)
    
    return fig

def _get_metric_name(metric: str) -> str:
    """Get display name for a metric"""
    names = {
        'sharpe': 'Sharpe Ratio',
        'sortino': 'Sortino Ratio',
        'volatility': 'Volatility (Annualized)',
        'drawdown': 'Maximum Drawdown',
        'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor',
        'avg_trade': 'Average Trade PnL ($)'
    }
    return names.get(metric, metric.replace('_', ' ').title())

def _get_metric_units(metric: str) -> str:
    """Get units for a metric"""
    units = {
        'sharpe': 'Ratio',
        'sortino': 'Ratio',
        'volatility': 'Volatility',
        'drawdown': 'Drawdown',
        'win_rate': 'Win Rate (%)',
        'profit_factor': 'Ratio',
        'avg_trade': 'PnL ($)'
    }
    return units.get(metric, '')

def _get_metric_color(metric: str, visualizer: BacktestVisualizer) -> str:
    """Get color for a metric"""
    colors = {
        'sharpe': visualizer.colors['up'],
        'sortino': visualizer.colors['up'],
        'volatility': visualizer.colors['down'],
        'drawdown': visualizer.colors['down'],
        'win_rate': visualizer.colors['up'],
        'profit_factor': visualizer.colors['up'],
        'avg_trade': visualizer.colors['highlight']
    }
    return colors.get(metric, visualizer.colors['neutral'])

def _create_empty_plot(message: str) -> go.Figure:
    """Create an empty plot with a message"""
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
        ],
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50),
        height=400
    )
    return fig
