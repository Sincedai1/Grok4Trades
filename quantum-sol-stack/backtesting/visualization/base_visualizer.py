"""
Base visualizer class for backtesting results
"""

from typing import Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from ..data_structures import BacktestResult

logger = logging.getLogger(__name__)

class BacktestVisualizer:
    """
    Base class for creating visualizations of backtest results
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', template: str = 'plotly_white'):
        """
        Initialize the visualizer
        
        Args:
            style: Matplotlib style to use
            template: Plotly template to use
        """
        self.style = style
        self.template = template
        
        # Set default styles
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Color palette
        self.colors = {
            'up': '#26a69a',  # Teal
            'down': '#ef5350',  # Red
            'neutral': '#7f8c8d',  # Gray
            'highlight': '#3498db',  # Blue
            'background': '#f8f9fa',  # Light gray
            'text': '#2c3e50',  # Dark blue-gray
            'grid': '#e0e0e0'  # Light gray
        }
    
    def create_comprehensive_report(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Create a comprehensive visual report with all key metrics and charts
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary containing all visualizations
        """
        from .equity_curve import plot_equity_curve
        from .drawdown import plot_drawdown
        from .returns import plot_returns_distribution, plot_monthly_returns_heatmap
        from .trade_analysis import plot_trade_analysis, plot_trade_duration_histogram
        from .rolling_metrics import plot_rolling_metrics
        
        report = {}
        
        try:
            # Equity curve
            report['equity_curve'] = plot_equity_curve(result, self)
            
            # Drawdown
            report['drawdown'] = plot_drawdown(result, self)
            
            # Returns distribution
            report['returns_distribution'] = plot_returns_distribution(result, self)
            
            # Monthly returns heatmap
            report['monthly_returns'] = plot_monthly_returns_heatmap(result, self)
            
            # Trade analysis
            report['trade_analysis'] = plot_trade_analysis(result, self)
            
            # Rolling metrics
            report['rolling_metrics'] = plot_rolling_metrics(result, self)
            
            # Trade duration histogram
            report['trade_duration'] = plot_trade_duration_histogram(result, self)
            
            logger.info("Comprehensive report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            raise
        
        return report
