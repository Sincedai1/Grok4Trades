from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

@dataclass
class Position:
    """Data class representing an open position"""
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0

class BasicRiskManager:
    """
    Basic risk management implementation with essential safety limits.
    Enforces position sizing, stop-loss, take-profit, and daily loss limits.
    """
    
    def __init__(self, 
                 max_risk_per_trade: float = 0.02,  # 2% risk per trade
                 max_daily_loss: float = 0.05,      # 5% max daily loss
                 max_position_size: float = 0.1,    # 10% of portfolio
                 max_leverage: float = 2.0,         # 2x leverage max
                 position_timeout_hours: int = 24):  # Close positions after 24h
        
        # Validate inputs
        if not 0 < max_risk_per_trade <= 0.1:  # Max 10% risk per trade
            raise ValueError("max_risk_per_trade must be between 0 and 0.1")
        if not 0 < max_daily_loss <= 0.1:      # Max 10% daily loss
            raise ValueError("max_daily_loss must be between 0 and 0.1")
        if not 0 < max_position_size <= 0.5:   # Max 50% of portfolio in one position
            raise ValueError("max_position_size must be between 0 and 0.5")
        if not 1.0 <= max_leverage <= 10.0:    # 1x to 10x leverage
            raise ValueError("max_leverage must be between 1.0 and 10.0")
            
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.position_timeout = timedelta(hours=position_timeout_hours)
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_balance = 0.0
        self.last_reset_date = self._get_current_date()
        
        logger.info("Initialized BasicRiskManager with "
                   f"max_risk={max_risk_per_trade*100:.1f}%, "
                   f"max_daily_loss={max_daily_loss*100:.1f}%, "
                   f"max_pos={max_position_size*100:.1f}%, "
                   f"max_lev={max_leverage}x")
    
    def _get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format"""
        return datetime.utcnow().strftime('%Y-%m-%d')
    
    def _reset_daily_metrics(self):
        """Reset daily metrics at the start of a new trading day"""
        current_date = self._get_current_date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_start_balance = self._get_current_balance()
            self.last_reset_date = current_date
            logger.info("Daily metrics reset")
    
    def _get_current_balance(self) -> float:
        """Get current portfolio balance (simplified)"""
        # In a real implementation, this would fetch from the exchange
        return 1000.0  # Default starting balance
    
    def calculate_position_size(self, 
                              entry_price: float, 
                              stop_loss: float, 
                              account_balance: float,
                              confidence: float = 1.0) -> Tuple[float, Dict]:
        """
        Calculate position size based on risk parameters.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            account_balance: Current account balance
            confidence: Confidence level of the trade (0.0 to 1.0)
            
        Returns:
            Tuple of (position_size, risk_metrics)
        """
        self._reset_daily_metrics()
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            logger.warning("Invalid stop loss. Position size set to 0.")
            return 0.0, {"error": "Invalid stop loss price", "risk_per_share": 0.0}
        
        # Calculate position value based on risk parameters
        risk_amount = min(
            account_balance * self.max_risk_per_trade * confidence,  # Risk-based size
            account_balance * self.max_position_size                  # Max position size
        )
        
        # Calculate position size in base currency
        position_size = risk_amount / risk_per_share
        
        # Calculate position value and leverage
        position_value = position_size * entry_price
        leverage = position_value / account_balance
        
        # Apply leverage limit
        if leverage > self.max_leverage:
            position_size = (account_balance * self.max_leverage) / entry_price
            position_value = position_size * entry_price
            leverage = self.max_leverage
            logger.warning(f"Leverage capped at {self.max_leverage}x")
        
        # Calculate risk metrics
        risk_metrics = {
            "risk_per_share": risk_per_share,
            "risk_amount": risk_amount,
            "position_value": position_value,
            "leverage": leverage,
            "stop_loss": stop_loss,
            "take_profit": None,  # Will be set by the strategy
            "risk_reward_ratio": None  # Will be set by the strategy
        }
        
        logger.info(
            f"Calculated position size: {position_size:.8f} "
            f"(Risk: ${risk_amount:.2f}, {self.max_risk_per_trade*100:.1f}% of balance, "
            f"Leverage: {leverage:.2f}x)"
        )
        
        return position_size, risk_metrics
    
    def check_daily_limits(self) -> Tuple[bool, str]:
        """
        Check if daily loss limit has been reached.
        
        Returns:
            Tuple of (within_limits, message)
        """
        self._reset_daily_metrics()
        
        current_balance = self._get_current_balance()
        daily_pnl_pct = (current_balance - self.daily_start_balance) / self.daily_start_balance
        
        if daily_pnl_pct < -self.max_daily_loss:
            message = (
                f"Daily loss limit reached: {daily_pnl_pct*100:.2f}% "
                f"(Limit: {self.max_daily_loss*100:.2f}%)"
            )
            logger.error(message)
            return False, message
            
        return True, f"Daily P&L: {daily_pnl_pct*100:.2f}%"
    
    def update_position(self, 
                       symbol: str, 
                       size: float, 
                       price: float,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Position:
        """
        Update or create a position.
        
        Args:
            symbol: Trading pair symbol
            size: Position size (positive for long, negative for short)
            price: Current price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Updated or new Position object
        """
        if symbol in self.positions:
            # Update existing position
            position = self.positions[symbol]
            position.size += size
            position.current_price = price
            
            # Update stop loss and take profit if provided
            if stop_loss is not None:
                position.stop_loss = stop_loss
            if take_profit is not None:
                position.take_profit = take_profit
                
            # Calculate P&L
            if position.size == 0:
                # Position closed
                pnl = (price - position.entry_price) * position.size
                pnl_pct = (pnl / (position.entry_price * abs(position.size))) * 100
                
                # Update daily P&L
                self.daily_pnl += pnl
                self.daily_trades += 1
                
                logger.info(
                    f"Position closed: {symbol} | "
                    f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | "
                    f"Daily P&L: ${self.daily_pnl:.2f}"
                )
                
                # Remove position if fully closed
                del self.positions[symbol]
                return position
            
            return position
        else:
            # Create new position
            if size == 0:
                raise ValueError("Cannot create a position with size 0")
                
            position = Position(
                symbol=symbol,
                size=size,
                entry_price=price,
                entry_time=datetime.utcnow(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=price
            )
            
            self.positions[symbol] = position
            logger.info(f"New position: {symbol} | Size: {size} | Entry: ${price:.2f}")
            return position
    
    def check_position_risk(self, symbol: str, current_price: float) -> Dict:
        """
        Check if a position should be closed based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            Dict with action and reason if position should be closed, None otherwise
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        position.current_price = current_price
        
        # Check if position has timed out
        time_in_trade = datetime.utcnow() - position.entry_time
        if time_in_trade > self.position_timeout:
            return {
                "action": "close",
                "reason": f"Position timed out after {self.position_timeout}",
                "price": current_price
            }
        
        # Check stop loss
        if position.stop_loss is not None:
            if (position.size > 0 and current_price <= position.stop_loss) or \
               (position.size < 0 and current_price >= position.stop_loss):
                return {
                    "action": "close",
                    "reason": f"Stop loss hit at {position.stop_loss}",
                    "price": position.stop_loss
                }
        
        # Check take profit
        if position.take_profit is not None:
            if (position.size > 0 and current_price >= position.take_profit) or \
               (position.size < 0 and current_price <= position.take_profit):
                return {
                    "action": "close",
                    "reason": f"Take profit hit at {position.take_profit}",
                    "price": position.take_profit
                }
        
        return None
    
    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions with current P&L"""
        positions = []
        for symbol, position in self.positions.items():
            if position.current_price is None:
                continue
                
            pnl = (position.current_price - position.entry_price) * position.size
            pnl_pct = (pnl / (position.entry_price * abs(position.size))) * 100
            
            positions.append({
                "symbol": symbol,
                "size": position.size,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "entry_time": position.entry_time.isoformat(),
                "duration": str(datetime.utcnow() - position.entry_time)
            })
            
        return positions
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        open_positions = self.get_open_positions()
        total_pnl = sum(p["pnl"] for p in open_positions)
        
        return {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "open_positions": len(open_positions),
            "total_pnl": total_pnl,
            "max_daily_loss": self.max_daily_loss,
            "max_risk_per_trade": self.max_risk_per_trade,
            "max_position_size": self.max_position_size,
            "max_leverage": self.max_leverage
        }
