import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import redis
from loguru import logger

@dataclass
class TradeSignal:
    """Data class representing a trading signal"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().timestamp()
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp,
            'timestamp_iso': datetime.utcfromtimestamp(self.timestamp).isoformat()
        }

@dataclass
class PositionSizing:
    """Data class representing position sizing calculation"""
    symbol: str
    size: float  # Position size in base currency
    usd_value: float  # Position size in USD
    risk_percentage: float  # Risk as % of capital
    stop_loss_pct: float  # Stop loss as % from entry
    kelly_fraction: float  # Kelly criterion fraction (0-1)
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'size': self.size,
            'usd_value': self.usd_value,
            'risk_percentage': self.risk_percentage,
            'stop_loss_pct': self.stop_loss_pct,
            'kelly_fraction': self.kelly_fraction
        }

class RiskManager:
    """Dynamic risk management system using Kelly Criterion and volatility adjustments"""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 max_risk_per_trade: float = 0.02,  # 2% risk per trade
                 max_portfolio_risk: float = 0.1,   # 10% max portfolio risk
                 volatility_lookback: int = 20,     # 20 periods for volatility
                 correlation_lookback: int = 60,    # 60 periods for correlation
                 redis_url: str = "redis://redis-cache:6379"):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.volatility_lookback = volatility_lookback
        self.correlation_lookback = correlation_lookback
        self.redis_url = redis_url
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.open_positions = {}
        self.trade_history = []
        self.volatility_cache = {}
        self.correlation_cache = {}
        
        # Initialize logger
        self.logger = logger.bind(module="RiskManager")
    
    def update_capital(self, new_capital: float):
        """Update the current trading capital"""
        self.current_capital = new_capital
        self._update_redis_metrics()
    
    def _update_redis_metrics(self):
        """Update risk metrics in Redis for monitoring"""
        try:
            metrics = {
                'current_capital': self.current_capital,
                'open_positions': len(self.open_positions),
                'max_drawdown': self._calculate_max_drawdown(),
                'timestamp': datetime.utcnow().isoformat()
            }
            self.redis_client.set('risk:metrics', json.dumps(metrics))
        except Exception as e:
            self.logger.error(f"Error updating Redis metrics: {e}")
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history"""
        if not self.trade_history:
            return 0.0
            
        equity_curve = [self.initial_capital]
        for trade in self.trade_history:
            if 'pnl_pct' in trade:
                equity_curve.append(equity_curve[-1] * (1 + trade['pnl_pct']))
        
        peak = equity_curve[0]
        max_drawdown = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown
    
    async def calculate_position_size(self, 
                                   signal: TradeSignal,
                                   price_data: pd.DataFrame) -> PositionSizing:
        """Calculate optimal position size using Kelly Criterion and volatility adjustments"""
        try:
            # Get recent price data
            returns = price_data['close'].pct_change().dropna()
            if len(returns) < 5:  # Not enough data
                return None
            
            # Calculate volatility (annualized)
            volatility = self._calculate_volatility(returns)
            
            # Calculate win rate and win/loss ratio (simplified)
            win_rate, win_loss_ratio = self._estimate_win_stats(signal.symbol)
            
            # Kelly fraction (simplified)
            kelly_fraction = self._kelly_criterion(win_rate, win_loss_ratio)
            
            # Adjust position size based on volatility
            vol_adjustment = self._volatility_adjustment(volatility)
            
            # Adjust for correlation with existing positions
            corr_adjustment = await self._correlation_adjustment(signal.symbol, price_data)
            
            # Calculate position size
            risk_capital = self.current_capital * self.max_risk_per_trade
            risk_capital *= vol_adjustment * corr_adjustment
            
            # Stop loss distance (as percentage of entry price)
            stop_loss_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            
            # Position size in base currency
            position_size = (risk_capital / stop_loss_pct) / signal.entry_price
            
            # Convert to USD value
            usd_value = position_size * signal.entry_price
            
            # Ensure position size doesn't exceed max portfolio risk
            max_position_usd = self.current_capital * self.max_portfolio_risk
            if usd_value > max_position_usd:
                usd_value = max_position_usd
                position_size = max_position_usd / signal.entry_price
            
            return PositionSizing(
                symbol=signal.symbol,
                size=position_size,
                usd_value=usd_value,
                risk_percentage=(usd_value / self.current_capital) * 100,
                stop_loss_pct=stop_loss_pct * 100,
                kelly_fraction=kelly_fraction
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return None
    
    def _calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility from returns"""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # Annualize with 252 trading days
        return vol
    
    def _estimate_win_stats(self, symbol: str) -> Tuple[float, float]:
        """Estimate win rate and win/loss ratio for a symbol"""
        # In a real implementation, this would use historical trade data
        # For now, return reasonable defaults
        return 0.55, 1.5  # 55% win rate, 1.5 win/loss ratio
    
    def _kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        if win_loss_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.1  # Default to 10% position size if inputs are invalid
            
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        # Use half-kelly for more conservative sizing
        return max(0.05, min(0.5, kelly / 2))
    
    def _volatility_adjustment(self, volatility: float) -> float:
        """Adjust position size based on volatility"""
        # Normalize volatility (assuming 20% annualized as baseline)
        normalized_vol = 0.20 / max(0.05, volatility)  # Cap at 5% minimum vol
        return min(1.5, max(0.5, normalized_vol))  # Bound between 0.5x and 1.5x
    
    async def _correlation_adjustment(self, symbol: str, price_data: pd.DataFrame) -> float:
        """Adjust position size based on correlation with existing positions"""
        if not self.open_positions:
            return 1.0  # No adjustment if no open positions
            
        # In a real implementation, calculate correlation with other positions
        # For now, return a fixed adjustment
        return 0.8  # 20% reduction for correlation
    
    def update_trade_outcome(self, 
                           symbol: str, 
                           pnl_pct: float, 
                           exit_price: float, 
                           exit_reason: str = None):
        """Update trade history with outcome of a closed position"""
        trade = {
            'symbol': symbol,
            'exit_price': exit_price,
            'exit_time': datetime.utcnow().isoformat(),
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason
        }
        
        self.trade_history.append(trade)
        
        # Remove from open positions if it exists
        if symbol in self.open_positions:
            del self.open_positions[symbol]
        
        # Update metrics in Redis
        self._update_redis_metrics()
        
        self.logger.info(f"Trade closed: {symbol} PnL: {pnl_pct:.2f}%")
    
    def get_risk_metrics(self) -> dict:
        """Get current risk metrics"""
        return {
            'current_capital': self.current_capital,
            'open_positions': len(self.open_positions),
            'max_drawdown': self._calculate_max_drawdown(),
            'total_trades': len(self.trade_history),
            'win_rate': self._calculate_win_rate(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0
            
        winning_trades = sum(1 for t in self.trade_history if t.get('pnl_pct', 0) > 0)
        return winning_trades / len(self.trade_history)

# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Sample price data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    price_data = pd.DataFrame({'close': prices}, index=dates)
    
    # Create risk manager
    risk_manager = RiskManager(initial_capital=10000.0)
    
    # Create a sample trade signal
    signal = TradeSignal(
        symbol="BTC/USDT",
        direction="LONG",
        entry_price=50000.0,
        stop_loss=48000.0
    )
    
    # Calculate position size
    position = risk_manager.calculate_position_size(signal, price_data)
    if position:
        print(f"Position size: {position.size:.6f} {signal.symbol.split('/')[0]}")
        print(f"USD value: ${position.usd_value:,.2f}")
        print(f"Risk percentage: {position.risk_percentage:.2f}%")
        print(f"Kelly fraction: {position.kelly_fraction:.2f}")
    
    # Simulate closing the trade
    risk_manager.update_trade_outcome(
        symbol="BTC/USDT",
        pnl_pct=2.5,  # 2.5% profit
        exit_price=51250.0,
        exit_reason="Take profit"
    )
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics()
    print("\nRisk metrics:", json.dumps(metrics, indent=2))
