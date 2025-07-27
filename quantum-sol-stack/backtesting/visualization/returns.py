"""
Returns visualization for backtesting results
"""

from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px

from ..data_structures import BacktestResult
from .base_visualizer import BacktestVisualizer

def plot_returns_distribution(
    result: BacktestResult, 
    visualizer: BacktestVisualizer,
    bins: int = 50,
    **kwargs
) -> go.Figure:
    """
    Plot distribution of returns with statistical information
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        bins: Number of bins for histogram
        
    Returns:
        Plotly Figure object
    """
    # Get returns data
    portfolio_df = result.portfolio_history.copy()
    
    # Calculate returns if not already present
    if 'returns' not in portfolio_df.columns:
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change().fillna(0)
    
    returns = portfolio_df['returns']
    
    # Create figure with distribution plot and Q-Q plot
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(
            'Returns Distribution', 
            'Q-Q Plot vs Normal Distribution'
        ),
        horizontal_spacing=0.15
    )
    
    # Histogram of returns
    fig.add_trace(
        go.Histogram(
            x=returns * 100,  # Convert to percentage
            nbinsx=bins,
            name='Returns',
            marker_color=visualizer.colors['highlight'],
            opacity=0.75,
            hovertemplate='%{x:.2f}%<br>Count: %{y}<extra></extra>'
        ),
        row=1,
        col=1
    )
    
    # Add KDE curve
    x_hist = np.linspace(returns.min(), returns.max(), 1000)
    
    fig.add_trace(
        go.Scatter(
            x=x_hist * 100,  # Convert to percentage
            y=np.histogram(returns, bins=100, density=True)[0] * 100,  # Scale to match histogram
            mode='lines',
            line=dict(color=visualizer.colors['down'], width=2),
            name='Density',
            yaxis='y2',
            hovertemplate='%{x:.2f}%<extra></extra>'
        ),
        row=1,
        col=1
    )
    
    # Q-Q Plot
    from scipy import stats
    
    # Calculate theoretical quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist='norm', fit=True)
    
    # Add Q-Q line
    line_x = np.array([osm[0], osm[-1]])
    line_y = intercept + slope * line_x
    
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            line=dict(color=visualizer.colors['neutral'], width=1.5, dash='dash'),
            name='Normal',
            hovertemplate='Theoretical Quantile: %{x:.2f}<br>Sample Quantile: %{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=1,
        col=2
    )
    
    # Add Q-Q points
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            marker=dict(
                color=visualizer.colors['highlight'],
                size=6,
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            name='Returns',
            hovertemplate='Theoretical Quantile: %{x:.2f}<br>Sample Quantile: %{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=1,
        col=2
    )
    
    # Update layout
    fig.update_layout(
        template=visualizer.template,
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50),
        height=450
    )
    
    # Update x-axes
    fig.update_xaxes(title_text='Return (%)', row=1, col=1)
    fig.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
    
    # Update y-axes
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Sample Quantiles', row=1, col=2)
    
    # Add statistical annotations
    if len(returns) > 0:
        from scipy import stats
        
        # Calculate statistics
        stats_text = [
            f'<b>Observations:</b> {len(returns):,}',
            f'<b>Mean:</b> {returns.mean()*100:.2f}%',
            f'<b>Median:</b> {returns.median()*100:.2f}%',
            f'<b>Std Dev:</b> {returns.std()*100:.2f}%',
            f'<b>Skewness:</b> {returns.skew():.2f}',
            f'<b>Kurtosis:</b> {returns.kurtosis():.2f}',
            f'<b>Jarque-Bera:</b> {stats.jarque_bera(returns)[0]:.2f} (p={stats.jarque_bera(returns)[1]:.4f})'
        ]
        
        # Add annotations
        annotations = []
        for i, text in enumerate(stats_text):
            annotations.append(
                dict(
                    x=0.02,
                    y=0.95 - i*0.05,
                    xref='paper',
                    yref='paper',
                    text=text,
                    showarrow=False,
                    font=dict(size=10),
                    align='left',
                    bgcolor='white',
                    bordercolor='lightgray',
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        fig.update_layout(annotations=annotations)
    
    return fig

def plot_monthly_returns_heatmap(
    result: BacktestResult,
    visualizer: BacktestVisualizer,
    **kwargs
) -> go.Figure:
    """
    Plot monthly returns heatmap
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        
    Returns:
        Plotly Figure object with heatmap
    """
    # Get returns data
    portfolio_df = result.portfolio_history.copy()
    
    # Calculate returns if not already present
    if 'returns' not in portfolio_df.columns:
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change().fillna(0)
    
    # Create a copy of the index to avoid SettingWithCopyWarning
    df = portfolio_df[['returns']].copy()
    
    # Extract year and month
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['month_name'] = df.index.month_name()
    
    # Pivot to create year-month matrix
    monthly_returns = df.pivot_table(
        index='year',
        columns='month',
        values='returns',
        aggfunc=lambda x: (1 + x).prod() - 1  # Compound monthly returns
    )
    
    # Replace column numbers with month names
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    monthly_returns.columns = [month_names[i-1] for i in monthly_returns.columns]
    
    # Create heatmap
    fig = go.Figure()
    
    # Create custom colorscale
    colorscale = [
        [0.0, visualizer.colors['down']],  # Red for negative returns
        [0.5, 'white'],  # White for zero
        [1.0, visualizer.colors['up']]  # Green for positive returns
    ]
    
    # Add heatmap trace
    fig.add_trace(
        go.Heatmap(
            z=monthly_returns.values * 100,  # Convert to percentage
            x=monthly_returns.columns,
            y=monthly_returns.index.astype(str),
            colorscale=colorscale,
            zmid=0,  # Center the colorscale at zero
            colorbar=dict(
                title='Return (%)',
                titleside='right',
                titlefont=dict(size=10),
                tickfont=dict(size=9),
                y=0.5,
                ypad=0
            ),
            hovertemplate=
                'Year: %{y}<br>' +
                'Month: %{x}<br>' +
                'Return: %{z:.2f}%<extra></extra>',
            hoverongaps=False
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Monthly Returns Heatmap',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title='Month',
        yaxis_title='Year',
        template=visualizer.template,
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50),
        height=400,
        width=800
    )
    
    # Add annotation for best and worst months
    if not monthly_returns.empty:
        max_return = monthly_returns.max().max()
        min_return = monthly_returns.min().min()
        
        if not np.isnan(max_return):
            max_idx = monthly_returns.stack().idxmax()
            fig.add_annotation(
                x=monthly_returns.columns.get_loc(max_idx[1]),
                y=monthly_returns.index.get_loc(max_idx[0]),
                text=f'{max_return*100:.1f}%',
                showarrow=False,
                font=dict(size=9, color='black')
            )
        
        if not np.isnan(min_return):
            min_idx = monthly_returns.stack().idxmin()
            fig.add_annotation(
                x=monthly_returns.columns.get_loc(min_idx[1]),
                y=monthly_returns.index.get_loc(min_idx[0]),
                text=f'{min_return*100:.1f}%',
                showarrow=False,
                font=dict(size=9, color='black')
            )
    
    return fig

def plot_returns_series(
    result: BacktestResult,
    visualizer: BacktestVisualizer,
    benchmark_col: str = 'benchmark',
    **kwargs
) -> go.Figure:
    """
    Plot cumulative returns over time with benchmark comparison
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        benchmark_col: Name of benchmark column in portfolio history
        
    Returns:
        Plotly Figure object
    """
    # Get returns data
    portfolio_df = result.portfolio_history.copy()
    
    # Calculate cumulative returns if not already present
    if 'cumulative_returns' not in portfolio_df.columns:
        if 'returns' not in portfolio_df.columns:
            portfolio_df['returns'] = portfolio_df['total_value'].pct_change().fillna(0)
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
    
    # Create figure
    fig = go.Figure()
    
    # Add strategy cumulative returns
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['cumulative_returns'] * 100,  # Convert to percentage
            mode='lines',
            name='Strategy',
            line=dict(color=visualizer.colors['highlight'], width=2),
            hovertemplate='%{y:.2f}%<extra></extra>'
        )
    )
    
    # Add benchmark if available
    if benchmark_col in portfolio_df.columns:
        # Calculate benchmark cumulative returns
        if f'{benchmark_col}_cumulative' not in portfolio_df.columns:
            benchmark_returns = portfolio_df[benchmark_col].pct_change().fillna(0)
            portfolio_df[f'{benchmark_col}_cumulative'] = (1 + benchmark_returns).cumprod() - 1
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df[f'{benchmark_col}_cumulative'] * 100,  # Convert to percentage
                mode='lines',
                name='Benchmark',
                line=dict(color=visualizer.colors['neutral'], width=1.5, dash='dash'),
                hovertemplate='%{y:.2f}%<extra></extra>'
            )
        )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='gray',
        opacity=0.5
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Cumulative Returns',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template=visualizer.template,
        hovermode='x',
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
    
    return fig
