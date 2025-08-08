from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import json

@dataclass
class Position:
    """Data class representing an open position"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float
    timestamp: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict] = None

class SimpleRiskManager:
    """
    Simple Risk Manager for position sizing and risk management
    
    This class handles position sizing, stop-loss/take-profit calculation,
    and risk management based on account balance and risk parameters.
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = 0.02,  # 2% risk per trade
        max_portfolio_risk: float = 0.1,   # 10% max portfolio risk
        max_daily_loss: float = 0.05,      # 5% max daily loss
        position_size_mode: str = 'fixed_fraction',  # 'fixed_fraction' or 'kelly'
        volatility_lookback: int = 20,     # Lookback period for volatility
        max_leverage: float = 10.0         # Maximum allowed leverage
    ):
        """
        Initialize the risk manager
        
        Parameters:
        -----------
        max_risk_per_trade : float, optional
            Maximum percentage of account to risk per trade (default: 0.02 for 2%)
        max_portfolio_risk : float, optional
            Maximum percentage of account at risk across all positions (default: 0.1 for 10%)
        max_daily_loss : float, optional
            Maximum percentage of account that can be lost in a day (default: 0.05 for 5%)
        position_size_mode : str, optional
            Position sizing method ('fixed_fraction' or 'kelly')
        volatility_lookback : int, optional
            Number of periods to calculate volatility (default: 20)
        max_leverage : float, optional
            Maximum allowed leverage (default: 10.0)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_daily_loss = max_daily_loss
        self.position_size_mode = position_size_mode
        self.volatility_lookback = volatility_lookback
        self.max_leverage = max_leverage
        
        # Track daily metrics
        self.daily_pnl = 0.0
        self.daily_high_balance = 0.0
        self.daily_low_balance = float('inf')
        self.last_reset = datetime.utcnow()
        
        # Track open positions
        self.positions: Dict[str, Position] = {}
        
        logger.info(f"Initialized Risk Manager with max_risk_per_trade={max_risk_per_trade*100}%")
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        confidence: float = 0.5,
        volatility: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate position size based on risk parameters
        
        Parameters:
        -----------
        account_balance : float
            Current account balance in quote currency
        entry_price : float
            Entry price for the position
        stop_loss : float, optional
            Stop loss price (required for fixed_fraction mode)
        confidence : float, optional
            Confidence in the trade (0.0 to 1.0)
        volatility : float, optional
            Volatility of the asset (annualized)
            
        Returns:
        --------
        Tuple[float, Dict]
            Position size in base currency and metadata
        """
        # Reset daily metrics if needed
        self._reset_daily_metrics_if_needed(account_balance)
        
        # Calculate available risk capital
        risk_capital = account_balance * self.max_risk_per_trade
        
        # Adjust risk based on confidence
        risk_capital *= confidence
        
        # Calculate position size based on selected mode
        if self.position_size_mode == 'fixed_fraction':
            if stop_loss is None:
                raise ValueError("stop_loss is required for fixed_fraction position sizing")
                
            # Calculate position size based on stop loss
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit <= 0:
                logger.warning("Invalid stop loss, using minimum position size")
                return 0.0, {'error': 'invalid_stop_loss'}
                
            position_size = risk_capital / risk_per_unit
            
            # Calculate position value and leverage
            position_value = position_size * entry_price
            leverage = position_value / account_balance
            
            # Apply maximum leverage constraint
            if leverage > self.max_leverage:
                position_size = (account_balance * self.max_leverage) / entry_price
                logger.warning(f"Leverage capped at {self.max_leverage}x")
            
            return position_size, {
                'position_size': position_size,
                'position_value': position_size * entry_price,
                'leverage': min(leverage, self.max_leverage),
                'risk_per_share': risk_per_unit,
                'risk_capital': risk_capital,
                'method': 'fixed_fraction'
            }
            
        elif self.position_size_mode == 'kelly':
            if volatility is None:
                raise ValueError("volatility is required for kelly position sizing")
                
            # Simple Kelly Criterion: f* = (p * (1 + r) - 1) / r
            # Where p is win probability, r is win/loss ratio
            # For simplicity, we'll use a simplified version
            
            # Base win probability on confidence (simplified)
            win_probability = 0.5 + (confidence - 0.5) * 0.5  # 0.5 to 0.75 range
            
            # Assume 1:2 risk:reward ratio (can be adjusted based on strategy)
            win_loss_ratio = 2.0
            
            # Kelly fraction
            kelly_fraction = (win_probability * (1 + win_loss_ratio) - 1) / win_loss_ratio
            
            # Conservative Kelly (half-Kelly)
            position_fraction = (kelly_fraction * 0.5) * (volatility / 0.2)  # Normalize by volatility
            
            # Calculate position size
            position_size = (account_balance * position_fraction) / entry_price
            
            # Calculate position value and leverage
            position_value = position_size * entry_price
            leverage = position_value / account_balance
            
            # Apply maximum leverage constraint
            if leverage > self.max_leverage:
                position_size = (account_balance * self.max_leverage) / entry_price
                logger.warning(f"Leverage capped at {self.max_leverage}x")
            
            return position_size, {
                'position_size': position_size,
                'position_value': position_size * entry_price,
                'leverage': min(leverage, self.max_leverage),
                'kelly_fraction': kelly_fraction,
                'win_probability': win_probability,
                'win_loss_ratio': win_loss_ratio,
                'method': 'kelly'
            }
            
        else:
            raise ValueError(f"Unsupported position size mode: {self.position_size_mode}")
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0,
        support_resistance: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price
        
        Parameters:
        -----------
        entry_price : float
            Entry price for the position
        atr : float, optional
            Average True Range for volatility-based stop
        atr_multiplier : float, optional
            Multiplier for ATR-based stop (default: 2.0)
        support_resistance : float, optional
            Support/resistance level for stop placement
            
        Returns:
        --------
        float
            Stop loss price
        """
        if support_resistance is not None:
            # Use support/resistance level if provided
            return support_resistance
        elif atr is not None:
            # Use ATR-based stop if ATR is provided
            return entry_price - (atr * atr_multiplier)
        else:
            # Default to 1% stop if nothing else is provided
            return entry_price * 0.99
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take profit price based on risk-reward ratio
        
        Parameters:
        -----------
        entry_price : float
            Entry price for the position
        stop_loss : float
            Stop loss price
        risk_reward_ratio : float, optional
            Desired risk-reward ratio (default: 2.0)
            
        Returns:
        --------
        float
            Take profit price
        """
        risk_amount = abs(entry_price - stop_loss)
        if entry_price > stop_loss:  # Long position
            return entry_price + (risk_amount * risk_reward_ratio)
        else:  # Short position
            return entry_price - (risk_amount * risk_reward_ratio)
    
    def update_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Update or add a position
        
        Parameters:
        -----------
        position_id : str
            Unique identifier for the position
        symbol : str
            Trading pair symbol (e.g., 'BTC/USDT')
        side : str
            Position side ('long' or 'short')
        entry_price : float
            Entry price
        size : float
            Position size in base currency
        stop_loss : float, optional
            Stop loss price
        take_profit : float, optional
            Take profit price
        metadata : dict, optional
            Additional position metadata
        """
        self.positions[position_id] = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            timestamp=datetime.utcnow().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )
        
        logger.info(f"Updated position {position_id}: {side} {size} {symbol} @ {entry_price}")
    
    def close_position(self, position_id: str) -> Optional[Position]:
        """
        Close a position
        
        Parameters:
        -----------
        position_id : str
            ID of the position to close
            
        Returns:
        --------
        Optional[Position]
            The closed position, or None if not found
        """
        if position_id in self.positions:
            position = self.positions.pop(position_id)
            logger.info(f"Closed position {position_id}")
            return position
        return None
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all open positions, optionally filtered by symbol
        
        Parameters:
        -----------
        symbol : str, optional
            Filter positions by symbol
            
        Returns:
        --------
        List[Position]
            List of open positions
        """
        if symbol:
            return [p for p in self.positions.values() if p.symbol == symbol]
        return list(self.positions.values())
    
    def update_pnl(self, pnl_delta: float) -> None:
        """
        Update P&L and check for daily loss limits
        
        Parameters:
        -----------
        pnl_delta : float
            Change in P&L (positive for profit, negative for loss)
            
        Returns:
        --------
        bool
            True if within daily loss limits, False if limit exceeded
        """
        self._reset_daily_metrics_if_needed()
        
        self.daily_pnl += pnl_delta
        
        # Update daily high/low
        self.daily_high_balance = max(self.daily_high_balance, self.daily_pnl)
        self.daily_low_balance = min(self.daily_low_balance, self.daily_pnl)
        
        # Check if daily loss limit is exceeded
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning(f"Daily loss limit exceeded: {self.daily_pnl:.2%}")
            return False
            
        return True
    
    def get_daily_metrics(self) -> Dict:
        """
        Get daily performance metrics
        
        Returns:
        --------
        Dict
            Dictionary containing daily metrics
        """
        self._reset_daily_metrics_if_needed()
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_high': self.daily_high_balance,
            'daily_low': self.daily_low_balance,
            'max_daily_loss': self.max_daily_loss,
            'last_reset': self.last_reset.isoformat(),
            'open_positions': len(self.positions)
        }
    
    def _reset_daily_metrics_if_needed(self, current_balance: Optional[float] = None) -> None:
        """
        Reset daily metrics if a new trading day has started
        
        Parameters:
        -----------
        current_balance : float, optional
            Current account balance to reset daily high/low
        """
        now = datetime.utcnow()
        
        # Check if we need to reset (new trading day)
        if now.date() > self.last_reset.date():
            logger.info("Resetting daily metrics for new trading day")
            
            # Reset metrics
            self.daily_pnl = 0.0
            self.daily_high_balance = current_balance if current_balance is not None else 0.0
            self.daily_low_balance = current_balance if current_balance is not None else float('inf')
            self.last_reset = now
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics
        
        Returns:
        --------
        Dict
            Dictionary containing current risk metrics
        """
        return {
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_daily_loss': self.max_daily_loss,
            'position_size_mode': self.position_size_mode,
            'volatility_lookback': self.volatility_lookback,
            'max_leverage': self.max_leverage,
            'open_positions_count': len(self.positions),
            'daily_metrics': self.get_daily_metrics()
        }
