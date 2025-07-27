"""
Equity curve visualization for backtesting results
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data_structures import BacktestResult
from .base_visualizer import BacktestVisualizer

def plot_equity_curve(
    result: BacktestResult, 
    visualizer: BacktestVisualizer,
    benchmark_col: str = 'benchmark',
    show_drawdown: bool = True,
    show_benchmark: bool = True,
    **kwargs
) -> go.Figure:
    """
    Plot equity curve with optional benchmark and drawdown
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        benchmark_col: Name of benchmark column in portfolio history
        show_drawdown: Whether to show drawdown in a subplot
        show_benchmark: Whether to show benchmark performance
        
    Returns:
        Plotly Figure object
    """
    # Get equity curve data
    portfolio_df = result.portfolio_history.copy()
    
    # Calculate returns if not already present
    if 'returns' not in portfolio_df.columns:
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change().fillna(0)
    
    # Create figure with optional subplots
    if show_drawdown:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Equity Curve', 'Drawdown')
        )
    else:
        fig = go.Figure()
    
    # Add equity curve trace
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['total_value'],
            name='Strategy',
            line=dict(color=visualizer.colors['highlight'], width=2),
            hovertemplate='%{y:$,.2f}<extra></extra>'
        ),
        row=1 if show_drawdown else None,
        col=1
    )
    
    # Add benchmark if available and requested
    if show_benchmark and benchmark_col in portfolio_df.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df[benchmark_col],
                name='Benchmark',
                line=dict(color=visualizer.colors['neutral'], width=1.5, dash='dash'),
                hovertemplate='%{y:$,.2f}<extra></extra>'
            ),
            row=1 if show_drawdown else None,
            col=1
        )
    
    # Add drawdown if requested
    if show_drawdown:
        # Calculate drawdown if not already in result
        if not hasattr(result, 'drawdown_series') or result.drawdown_series is None:
            from ..performance_metrics import PerformanceMetrics
            metrics = PerformanceMetrics()
            result.drawdown_series = metrics.calculate_drawdown_series(portfolio_df['total_value'])
        
        # Add drawdown area
        fig.add_trace(
            go.Scatter(
                x=result.drawdown_series.index,
                y=result.drawdown_series * 100,  # Convert to percentage
                fill='tozeroy',
                fillcolor=f"rgba(239, 83, 80, 0.3)",  # Semi-transparent red
                line=dict(color=visualizer.colors['down'], width=1),
                name='Drawdown',
                hovertemplate='%{y:.2f}%<extra></extra>',
                showlegend=False
            ),
            row=2,
            col=1
        )
    
    # Update layout
    fig.update_layout(
        template=visualizer.template,
        title=dict(
            text='Equity Curve',
            x=0.5,
            xanchor='center',
            y=0.95,
            font=dict(size=16)
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Update y-axes
    if show_drawdown:
        fig.update_yaxes(
            title_text='Portfolio Value ($)',
            row=1,
            col=1,
            gridcolor=visualizer.colors['grid'],
            zerolinecolor=visualizer.colors['grid']
        )
        fig.update_yaxes(
            title_text='Drawdown (%)',
            row=2,
            col=1,
            gridcolor=visualizer.colors['grid'],
            zerolinecolor=visualizer.colors['grid']
        )
        fig.update_xaxes(row=2, col=1, title_text='Date')
    else:
        fig.update_yaxes(title_text='Portfolio Value ($)')
        fig.update_xaxes(title_text='Date')
    
    # Add annotations for key metrics
    if len(portfolio_df) > 0:
        initial_value = portfolio_df['total_value'].iloc[0]
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        annotations = [
            dict(
                x=0.02,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'<b>Total Return:</b> {total_return:.2f}%',
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                x=0.02,
                y=0.90,
                xref='paper',
                yref='paper',
                text=f'<b>Final Value:</b> ${final_value:,.2f}',
                showarrow=False,
                font=dict(size=12)
            )
        ]
        
        if hasattr(result, 'performance_metrics'):
            metrics = result.performance_metrics
            annotations.extend([
                dict(
                    x=0.02,
                    y=0.85,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Sharpe:</b> {metrics.get("sharpe_ratio", 0):.2f}',
                    showarrow=False,
                    font=dict(size=12)
                ),
                dict(
                    x=0.02,
                    y=0.80,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Max DD:</b> {abs(metrics.get("max_drawdown", 0))*100:.2f}%',
                    showarrow=False,
                    font=dict(size=12)
                )
            ])
        
        fig.update_layout(annotations=annotations)
    
    return fig
