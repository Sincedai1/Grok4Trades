"""
Drawdown visualization for backtesting results
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data_structures import BacktestResult
from .base_visualizer import BacktestVisualizer

def plot_drawdown(
    result: BacktestResult, 
    visualizer: BacktestVisualizer,
    show_underwater: bool = True,
    show_periods: bool = True,
    **kwargs
) -> go.Figure:
    """
    Plot drawdown analysis with underwater chart and drawdown periods
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        show_underwater: Whether to show the underwater equity curve
        show_periods: Whether to highlight drawdown periods
        
    Returns:
        Plotly Figure object
    """
    # Get portfolio data
    portfolio_df = result.portfolio_history.copy()
    
    # Calculate drawdown series if not already in result
    if not hasattr(result, 'drawdown_series') or result.drawdown_series is None:
        from ..performance_metrics import PerformanceMetrics
        metrics = PerformanceMetrics()
        drawdown_series = metrics.calculate_drawdown_series(portfolio_df['total_value'])
    else:
        drawdown_series = result.drawdown_series
    
    # Create figure
    fig = go.Figure()
    
    # Add underwater area
    if show_underwater:
        fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series * 100,  # Convert to percentage
                fill='tozeroy',
                fillcolor=f"rgba(239, 83, 80, 0.3)",  # Semi-transparent red
                line=dict(color=visualizer.colors['down'], width=1.5),
                name='Drawdown',
                hovertemplate='%{y:.2f}%<extra></extra>'
            )
        )
    
    # Add drawdown periods if requested
    if show_periods and hasattr(result, 'drawdown_periods'):
        for period in result.drawdown_periods:
            fig.add_vrect(
                x0=period['start'],
                x1=period['end'],
                fillcolor=visualizer.colors['down'],
                opacity=0.1,
                layer="below",
                line_width=0,
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Underwater Equity Curve',
            x=0.5,
            xanchor='center',
            y=0.95,
            font=dict(size=16)
        ),
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template=visualizer.template,
        hovermode='x',
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False
    )
    
    # Add horizontal line at max drawdown
    if hasattr(result, 'performance_metrics') and 'max_drawdown' in result.performance_metrics:
        max_dd = result.performance_metrics['max_drawdown'] * 100  # to percentage
        
        fig.add_hline(
            y=-max_dd,
            line_dash='dash',
            line_color=visualizer.colors['down'],
            annotation_text=f'Max Drawdown: {-max_dd:.2f}%',
            annotation_position='bottom right'
        )
    
    # Add annotations for key metrics
    if hasattr(result, 'performance_metrics'):
        metrics = result.performance_metrics
        annotations = []
        
        if 'max_drawdown' in metrics:
            annotations.append(
                dict(
                    x=0.02,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Max Drawdown:</b> {abs(metrics["max_drawdown"])*100:.2f}%',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='white',
                    bordercolor='lightgray',
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        if 'avg_drawdown' in metrics:
            annotations.append(
                dict(
                    x=0.02,
                    y=0.88,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Avg Drawdown:</b> {abs(metrics["avg_drawdown"])*100:.2f}%',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='white',
                    bordercolor='lightgray',
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        if 'max_drawdown_duration' in metrics:
            annotations.append(
                dict(
                    x=0.02,
                    y=0.81,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Max DD Duration:</b> {metrics["max_drawdown_duration"]} days',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='white',
                    bordercolor='lightgray',
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        if 'calmar_ratio' in metrics:
            annotations.append(
                dict(
                    x=0.02,
                    y=0.74,
                    xref='paper',
                    yref='paper',
                    text=f'<b>Calmar Ratio:</b> {metrics["calmar_ratio"]:.2f}',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='white',
                    bordercolor='lightgray',
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        fig.update_layout(annotations=annotations)
    
    return fig

def plot_drawdown_periods(
    result: BacktestResult,
    visualizer: BacktestVisualizer,
    top_n: int = 5,
    **kwargs
) -> go.Figure:
    """
    Plot the top N largest drawdown periods
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        top_n: Number of largest drawdowns to show
        
    Returns:
        Plotly Figure object with horizontal bar chart
    """
    if not hasattr(result, 'drawdown_periods') or not result.drawdown_periods:
        # If no drawdown periods calculated, return empty figure
        fig = go.Figure()
        fig.update_layout(
            title='No Drawdown Periods',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text='No drawdown periods found',
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=16)
            )]
        )
        return fig
    
    # Sort drawdown periods by magnitude
    sorted_dd = sorted(result.drawdown_periods, key=lambda x: x['drawdown'])
    
    # Take top N largest drawdowns
    top_dd = sorted_dd[:min(top_n, len(sorted_dd))]
    
    # Prepare data for plotting
    dd_magnitudes = [abs(dd['drawdown']) * 100 for dd in top_dd]  # to percentage
    dd_dates = [
        f"{dd['start'].strftime('%Y-%m-%d')} to {dd['end'].strftime('%Y-%m-%d')} "
        f"({dd['days']} days)" 
        for dd in top_dd
    ]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            y=dd_dates,
            x=dd_magnitudes,
            orientation='h',
            marker_color=visualizer.colors['down'],
            opacity=0.7,
            hovertemplate='%{x:.2f}%<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Top {len(top_dd)} Largest Drawdown Periods',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title='Drawdown (%)',
        yaxis_title='Period',
        template=visualizer.template,
        height=200 + 30 * len(top_dd),  # Dynamic height based on number of bars
        margin=dict(t=50, b=50, l=150, r=50),
        showlegend=False
    )
    
    return fig
