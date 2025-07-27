"""
Trade analysis visualization for backtesting results
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from ..data_structures import BacktestResult
from ..portfolio_simulator import Trade, PortfolioSimulator as Portfolio
from .base_visualizer import BacktestVisualizer


def plot_trade_performance(
    trades: Union[List[Trade], pd.DataFrame],
    portfolio: Optional[Portfolio] = None,
    title: str = 'Trade Performance Analysis',
    **kwargs
) -> go.Figure:
    """
    Plot trade performance metrics including PnL distribution and statistics.
    
    Args:
        trades: List of Trade objects or DataFrame with trade data
        portfolio: Optional Portfolio object for additional metrics
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    # Handle empty trades
    if not trades:
        return _create_empty_plot('No trades to display')
    
    # Convert trades to DataFrame if needed
    if isinstance(trades, list):
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'symbol': t.symbol,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'side': t.side,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'duration': (t.exit_time - t.entry_time).total_seconds() / 3600  # in hours
        } for t in trades])
    else:
        trades_df = trades.copy()
    
    # Calculate metrics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Create subplots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            'PnL Distribution',
            'Win/Loss Metrics',
            'Trade Duration vs. PnL',
            'Daily PnL'
        ),
        specs=[[{"type": "histogram"}, {"type": "pie"}],
              [{"type": "scatter"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. PnL Distribution
    fig.add_trace(
        go.Histogram(
            x=trades_df['pnl'],
            name='PnL Distribution',
            nbinsx=50,
            marker_color='#636efa',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 2. Win/Loss Metrics
    win_loss_data = pd.Series({
        'Winning Trades': len(winning_trades),
        'Losing Trades': len(losing_trades),
        'Break-even': len(trades_df) - len(winning_trades) - len(losing_trades)
    })
    
    fig.add_trace(
        go.Pie(
            labels=win_loss_data.index,
            values=win_loss_data.values,
            name='Win/Loss',
            hole=0.5,
            textinfo='percent+label',
            marker=dict(colors=['#00cc96', '#ef553b', '#636efa'])
        ),
        row=1, col=2
    )
    
    # 3. Trade Duration vs. PnL
    fig.add_trace(
        go.Scatter(
            x=trades_df['duration'],
            y=trades_df['pnl'],
            mode='markers',
            name='Trades',
            text=trades_df['symbol'],
            marker=dict(
                color=np.where(trades_df['pnl'] > 0, '#00cc96', '#ef553b'),
                size=8,
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            )
        ),
        row=2, col=1
    )
    
    # 4. Daily PnL (if portfolio is provided)
    if portfolio is not None and hasattr(portfolio, 'history'):
        daily_pnl = portfolio.history.groupby(
            portfolio.history.index.date
        )['total_pnl'].sum()
        
        fig.add_trace(
            go.Bar(
                x=daily_pnl.index,
                y=daily_pnl.values,
                name='Daily PnL',
                marker_color=np.where(daily_pnl > 0, '#00cc96', '#ef553b')
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        height=800,
        margin=dict(t=80, b=50, l=50, r=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes
    fig.update_xaxes(title_text='PnL ($)', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    
    fig.update_xaxes(title_text='Win/Loss Distribution', row=1, col=2)
    
    fig.update_xaxes(title_text='Duration (hours)', row=2, col=1)
    fig.update_yaxes(title_text='PnL ($)', row=2, col=1)
    
    if portfolio is not None:
        fig.update_xaxes(title_text='Date', row=2, col=2)
        fig.update_yaxes(title_text='PnL ($)', row=2, col=2)
    
    return fig

def plot_trade_analysis(
    result: BacktestResult, 
    visualizer: BacktestVisualizer,
    **kwargs
) -> go.Figure:
    """
    Plot trade analysis including PnL distribution and win/loss metrics
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        
    Returns:
        Plotly Figure object
    """
    if not hasattr(result, 'trades') or not result.trades:
        return _create_empty_plot('No Trades Found')
    
    # Prepare trade data
    trades = result.trades
    pnls = [t.pnl for t in trades if t.pnl is not None]
    
    if not pnls:
        return _create_empty_plot('No Valid Trades')
    
    # Create subplots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            'Trade PnL Distribution',
            'Win/Loss Metrics',
            'Trade Duration vs. PnL',
            'Entry/Exit Analysis'
        ),
        specs=[[{"type": "xy"}, {"type": "domain"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # 1. PnL Distribution
    fig.add_trace(
        go.Histogram(
            x=pnls,
            nbinsx=50,
            name='PnL Distribution',
            marker_color=visualizer.colors['highlight'],
            opacity=0.7,
            hovertemplate='PnL: $%{x:,.2f}<br>Count: %{y}<extra></extra>'
        ),
        row=1,
        col=1
    )
    
    # Add mean line
    mean_pnl = np.mean(pnls)
    fig.add_vline(
        x=mean_pnl,
        line_dash='dash',
        line_color=visualizer.colors['down'],
        annotation_text=f'Mean: ${mean_pnl:,.2f}',
        annotation_position='top right',
        row=1,
        col=1
    )
    
    # 2. Win/Loss Metrics
    winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
    breakeven_trades = [t for t in trades if t.pnl == 0]
    
    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
    profit_factor = (sum(t.pnl for t in winning_trades) / 
                    abs(sum(t.pnl for t in losing_trades))) if losing_trades else float('inf')
    
    # Win/Loss Pie Chart
    labels = ['Winning Trades', 'Losing Trades', 'Breakeven Trades']
    values = [len(winning_trades), len(losing_trades), len(breakeven_trades)]
    
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker_colors=[
                visualizer.colors['up'],
                visualizer.colors['down'],
                visualizer.colors['neutral']
            ],
            hoverinfo='label+percent+value',
            textinfo='percent',
            textposition='inside'
        ),
        row=1,
        col=2
    )
    
    # 3. Trade Duration vs PnL
    durations = []
    trade_pnls = []
    
    for trade in trades:
        if trade.entry_time and trade.exit_time and trade.pnl is not None:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # in hours
            durations.append(duration)
            trade_pnls.append(trade.pnl)
    
    if durations and trade_pnls:
        fig.add_trace(
            go.Scatter(
                x=durations,
                y=trade_pnls,
                mode='markers',
                marker=dict(
                    color=[
                        visualizer.colors['up'] if pnl > 0 else visualizer.colors['down'] 
                        for pnl in trade_pnls
                    ],
                    size=8,
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name='Trade',
                hovertemplate='Duration: %{x:.1f} hours<br>PnL: $%{y:,.2f}<extra></extra>',
                showlegend=False
            ),
            row=2,
            col=1
        )
        
        # Add regression line
        if len(durations) > 1:
            z = np.polyfit(durations, trade_pnls, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=sorted(durations),
                    y=p(sorted(durations)),
                    mode='lines',
                    line=dict(color=visualizer.colors['neutral'], width=2, dash='dash'),
                    name='Trend',
                    showlegend=False
                ),
                row=2,
                col=1
            )
    
    # 4. Entry/Exit Analysis
    if hasattr(result, 'entry_exit_analysis'):
        entry_exit = result.entry_exit_analysis
        
        # Example: Hour of day analysis
        if 'hourly_returns' in entry_exit:
            hours = list(entry_exit['hourly_returns'].keys())
            avg_returns = list(entry_exit['hourly_returns'].values())
            
            fig.add_trace(
                go.Bar(
                    x=hours,
                    y=avg_returns,
                    name='Avg Return by Hour',
                    marker_color=visualizer.colors['highlight'],
                    opacity=0.7,
                    hovertemplate='Hour: %{x}<br>Avg Return: %{y:.2%}<extra></extra>'
                ),
                row=2,
                col=2
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Trade Analysis',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template=visualizer.template,
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(t=100, b=50, l=50, r=50),
        height=800
    )
    
    # Update subplot titles and axes
    fig.update_xaxes(title_text='PnL ($)', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    
    if durations and trade_pnls:
        fig.update_xaxes(title_text='Duration (hours)', row=2, col=1)
        fig.update_yaxes(title_text='PnL ($)', row=2, col=1)
    
    if 'hourly_returns' in locals() and 'entry_exit' in locals() and 'hourly_returns' in entry_exit:
        fig.update_xaxes(title_text='Hour of Day', row=2, col=2)
        fig.update_yaxes(title_text='Average Return', row=2, col=2, tickformat='.1%')
    
    # Add annotations with key metrics
    annotations = [
        dict(
            x=0.98,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f'<b>Win Rate:</b> {win_rate:.1%}<br>'
                 f'<b>Avg Win:</b> ${avg_win:,.2f}<br>'
                 f'<b>Avg Loss:</b> ${avg_loss:,.2f}<br>'
                 f'<b>Profit Factor:</b> {profit_factor:.2f}',
            showarrow=False,
            align='right',
            bordercolor='lightgray',
            borderwidth=1,
            borderpad=4,
            bgcolor='white',
            opacity=0.9
        )
    ]
    
    fig.update_layout(annotations=annotations)
    
    return fig

def plot_trade_duration_histogram(
    result: BacktestResult,
    visualizer: BacktestVisualizer,
    **kwargs
) -> go.Figure:
    """
    Plot histogram of trade durations
    
    Args:
        result: BacktestResult object
        visualizer: BacktestVisualizer instance
        
    Returns:
        Plotly Figure object
    """
    if not hasattr(result, 'trades') or not result.trades:
        return _create_empty_plot('No Trades Found')
    
    # Calculate trade durations in hours
    durations = []
    
    for trade in result.trades:
        if trade.entry_time and trade.exit_time:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # in hours
            durations.append(duration)
    
    if not durations:
        return _create_empty_plot('No Valid Trade Durations')
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=durations,
            nbinsx=50,
            name='Trade Duration',
            marker_color=visualizer.colors['highlight'],
            opacity=0.7,
            hovertemplate='Duration: %{x:.1f} hours<br>Count: %{y}<extra></extra>'
        )
    )
    
    # Add mean and median lines
    mean_duration = np.mean(durations)
    median_duration = np.median(durations)
    
    fig.add_vline(
        x=mean_duration,
        line_dash='dash',
        line_color=visualizer.colors['down'],
        annotation_text=f'Mean: {mean_duration:.1f}h',
        annotation_position='top right'
    )
    
    fig.add_vline(
        x=median_duration,
        line_dash='dot',
        line_color=visualizer.colors['up'],
        annotation_text=f'Median: {median_duration:.1f}h',
        annotation_position='top right'
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Trade Duration Distribution',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title='Duration (hours)',
        yaxis_title='Number of Trades',
        template=visualizer.template,
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50),
        height=400
    )
    
    # Add statistics annotations
    stats_text = [
        f'<b>Total Trades:</b> {len(durations):,}',
        f'<b>Min Duration:</b> {min(durations):.1f} hours',
        f'<b>Max Duration:</b> {max(durations):.1f} hours',
        f'<b>Mean Duration:</b> {mean_duration:.1f} hours',
        f'<b>Median Duration:</b> {median_duration:.1f} hours',
        f'<b>Std Dev:</b> {np.std(durations):.1f} hours'
    ]
    
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
