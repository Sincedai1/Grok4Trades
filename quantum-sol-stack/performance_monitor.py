import time
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import aiohttp
from dataclasses import dataclass, asdict
import redis
from loguru import logger

@dataclass
class TradeRecord:
    """Data class representing a trade record"""
    trade_id: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    size: float
    pnl: float
    pnl_pct: float
    fees: float
    strategy: str
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        """Convert trade record to dictionary"""
        result = asdict(self)
        result['entry_time_iso'] = datetime.utcfromtimestamp(self.entry_time).isoformat()
        result['exit_time_iso'] = datetime.utcfromtimestamp(self.exit_time).isoformat()
        result['duration_seconds'] = self.exit_time - self.entry_time
        result['duration_hours'] = (self.exit_time - self.entry_time) / 3600
        return result

class PerformanceMonitor:
    """Performance monitoring and analysis for trading strategies"""
    
    def __init__(self, 
                 redis_url: str = "redis://redis-cache:6379",
                 prometheus_url: str = "http://prometheus:9090",
                 metrics_prefix: str = "trading"):
        
        self.redis_url = redis_url
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prometheus_url = prometheus_url
        self.metrics_prefix = metrics_prefix
        self.trades: List[TradeRecord] = []
        self.metrics_cache = {}
        self.metrics_cache_ttl = 60  # 1 minute cache TTL
        self.metrics_last_updated = 0
        
        # Configure logger
        self.logger = logger.bind(module="PerformanceMonitor")
    
    async def record_trade(self, trade: TradeRecord):
        """Record a completed trade"""
        try:
            # Add to in-memory list
            self.trades.append(trade)
            
            # Add to Redis for persistence
            trade_key = f"{self.metrics_prefix}:trade:{trade.trade_id}"
            self.redis_client.set(trade_key, json.dumps(trade.to_dict()))
            
            # Add to time-series data
            ts = int(time.time() * 1000)  # Milliseconds precision
            self.redis_client.zadd(f"{self.metrics_prefix}:trades:all", {trade_key: ts})
            self.redis_client.zadd(f"{self.metrics_prefix}:trades:{trade.symbol}", {trade_key: ts})
            
            # Update metrics cache
            self._update_metrics_cache()
            
            self.logger.info(f"Recorded trade {trade.trade_id}: {trade.symbol} {trade.direction} PnL: {trade.pnl_pct:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def _update_metrics_cache(self):
        """Update cached metrics"""
        current_time = time.time()
        if current_time - self.metrics_last_updated < self.metrics_cache_ttl:
            return
            
        try:
            # Calculate performance metrics
            if not self.trades:
                return
                
            df = pd.DataFrame([t.to_dict() for t in self.trades])
            
            # Basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['pnl_pct'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = df['pnl'].sum()
            avg_pnl = df['pnl'].mean()
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(df)
            sharpe_ratio = self._calculate_sharpe_ratio(df)
            sortino_ratio = self._calculate_sortino_ratio(df)
            
            # Update cache
            self.metrics_cache = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'last_updated': current_time
            }
            
            # Update Redis
            self.redis_client.set(
                f"{self.metrics_prefix}:performance_metrics",
                json.dumps(self.metrics_cache)
            )
            
            self.metrics_last_updated = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating metrics cache: {e}")
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown from trade PnL"""
        if df.empty:
            return 0.0
            
        cum_returns = (1 + df['pnl_pct'] / 100).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min() * 100  # Return as percentage
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(df) < 2:
            return 0.0
            
        returns = df['pnl_pct'] / 100  # Convert to decimal
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
            
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sortino ratio"""
        if len(df) < 2:
            return 0.0
            
        returns = df['pnl_pct'] / 100  # Convert to decimal
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0.0
            
        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    
    async def get_performance_metrics(self, use_cache: bool = True) -> dict:
        """Get current performance metrics"""
        if use_cache and self.metrics_cache:
            return self.metrics_cache
            
        try:
            # Try to get from Redis first
            cached_metrics = self.redis_client.get(f"{self.metrics_prefix}:performance_metrics")
            if cached_metrics:
                self.metrics_cache = json.loads(cached_metrics)
                self.metrics_last_updated = time.time()
                return self.metrics_cache
                
            # If not in cache, recalculate
            self._update_metrics_cache()
            return self.metrics_cache
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def detect_performance_degradation(self, 
                                          lookback_days: int = 30,
                                          threshold_pct: float = -20.0) -> Tuple[bool, str]:
        """Detect if strategy performance has degraded"""
        try:
            # Get recent trades
            cutoff_time = (datetime.now() - timedelta(days=lookback_days)).timestamp()
            recent_trades = [t for t in self.trades if t.exit_time >= cutoff_time]
            
            if len(recent_trades) < 5:  # Not enough data
                return False, "Insufficient data for performance analysis"
            
            # Calculate metrics for recent period
            recent_df = pd.DataFrame([t.to_dict() for t in recent_trades])
            recent_pnl_pct = recent_df['pnl_pct'].sum()
            
            # Compare with historical performance
            hist_df = pd.DataFrame([t.to_dict() for t in self.trades if t.exit_time < cutoff_time])
            if hist_df.empty:
                return False, "No historical data for comparison"
                
            hist_pnl_pct = hist_df['pnl_pct'].mean() * len(recent_trades)  # Scale to same number of trades
            
            # Calculate performance change
            pct_change = ((recent_pnl_pct - hist_pnl_pct) / abs(hist_pnl_pct)) * 100
            
            if pct_change < threshold_pct:
                return True, (
                    f"Performance degraded by {abs(pct_change):.1f}% in the last {lookback_days} days. "
                    f"Recent PnL: {recent_pnl_pct:.2f}% vs Historical: {hist_pnl_pct:.2f}%"
                )
            
            return False, f"Performance within normal range. Recent PnL: {recent_pnl_pct:.2f}%"
            
        except Exception as e:
            self.logger.error(f"Error detecting performance degradation: {e}")
            return False, f"Error analyzing performance: {str(e)}"
    
    async def generate_report(self, 
                            output_format: str = 'json',
                            save_to_file: bool = False) -> str:
        """Generate a performance report"""
        try:
            metrics = await self.get_performance_metrics()
            
            if output_format.lower() == 'json':
                report = json.dumps(metrics, indent=2)
            else:
                # Generate text report
                report = f"""
                Performance Report
                ==================
                
                Summary
                -------
                Total Trades: {total_trades}
                Win Rate: {win_rate:.1f}%
                Total PnL: ${total_pnl:,.2f}
                
                Averages
                --------
                Avg PnL: ${avg_pnl:,.2f}
                Avg Win: ${avg_win:,.2f}
                Avg Loss: ${avg_loss:,.2f}
                
                Risk Metrics
                -----------
                Max Drawdown: {max_drawdown:.2f}%
                Sharpe Ratio: {sharpe_ratio:.2f}
                Sortino Ratio: {sortino_ratio:.2f}
                Profit Factor: {profit_factor:.2f}
                
                Last Updated: {timestamp}
                """.format(
                    total_trades=metrics.get('total_trades', 0),
                    win_rate=metrics.get('win_rate', 0),
                    total_pnl=metrics.get('total_pnl', 0),
                    avg_pnl=metrics.get('avg_pnl', 0),
                    avg_win=metrics.get('avg_win', 0),
                    avg_loss=metrics.get('avg_loss', 0),
                    max_drawdown=metrics.get('max_drawdown', 0),
                    sharpe_ratio=metrics.get('sharpe_ratio', 0),
                    sortino_ratio=metrics.get('sortino_ratio', 0),
                    profit_factor=metrics.get('profit_factor', 0),
                    timestamp=metrics.get('timestamp', 'N/A')
                )
            
            if save_to_file:
                filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
                with open(filename, 'w') as f:
                    f.write(report)
                self.logger.info(f"Report saved to {filename}")
            
            return report
            
        except Exception as e:
            error_msg = f"Error generating performance report: {e}"
            self.logger.error(error_msg)
            return error_msg

# Example usage
if __name__ == "__main__":
    import random
    from datetime import datetime, timedelta
    
    async def main():
        # Create performance monitor
        monitor = PerformanceMonitor()
        
        # Generate some sample trades
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        start_time = int((datetime.now() - timedelta(days=60)).timestamp())
        
        for i in range(100):
            trade = TradeRecord(
                trade_id=f"trade_{i+1}",
                symbol=random.choice(symbols),
                direction=random.choice(['LONG', 'SHORT']),
                entry_price=random.uniform(100, 50000),
                exit_price=random.uniform(80, 52000),
                entry_time=start_time + (i * 3600 * 12),  # One trade every 12 hours
                exit_time=start_time + ((i + 1) * 3600 * 10),  # 10 hours later
                size=random.uniform(0.1, 10),
                pnl=random.uniform(-1000, 1500),
                pnl_pct=random.uniform(-5, 8),
                fees=random.uniform(0.1, 10),
                strategy=f"strategy_{random.randint(1, 3)}",
                tags=[f"tag_{random.randint(1, 5)}" for _ in range(2)]
            )
            await monitor.record_trade(trade)
        
        # Get performance metrics
        metrics = await monitor.get_performance_metrics()
        print("\nPerformance Metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Check for performance degradation
        degraded, message = await monitor.detect_performance_degradation()
        print(f"\nPerformance Degradation Detected: {degraded}")
        print(f"Message: {message}")
        
        # Generate a report
        report = await monitor.generate_report(output_format='text')
        print("\nPerformance Report:")
        print(report)
    
    asyncio.run(main())
