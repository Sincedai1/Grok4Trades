"""
Portfolio Simulator for QuantumSol Backtesting Framework

This module provides a comprehensive portfolio simulation engine that handles:
- Position management with support for long/short positions
- Realistic trade execution with configurable fees and slippage
- Risk management with position sizing and drawdown limits
- Multi-currency support with FX conversion
- Performance tracking and reporting
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeDirection(Enum):
    """Enum for trade directions."""
    LONG = auto()
    SHORT = auto()

class OrderType(Enum):
    """Enum for order types."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()

@dataclass
class Position:
    """
    Represents an open position in the portfolio.
    
    Attributes:
        symbol: Trading symbol (e.g., 'BTC/USD')
        entry_price: Entry price of the position
        size: Position size in base currency
        direction: TradeDirection.LONG or TradeDirection.SHORT
        entry_time: Timestamp when position was opened
        current_price: Current market price
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        unrealized_pnl: Current unrealized P&L
        unrealized_pnl_pct: Current unrealized P&L as percentage
        metadata: Additional position metadata
    """
    symbol: str
    entry_price: float
    size: float
    direction: TradeDirection
    entry_time: datetime
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, price: float) -> None:
        """
        Update position with current market price and recalculate P&L.
        
        Args:
            price: Current market price
        """
        self.current_price = price
        if self.direction == TradeDirection.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.size
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.size
        
        if self.entry_price > 0:  # Avoid division by zero
            self.unrealized_pnl_pct = (self.unrealized_pnl / 
                                     (self.entry_price * self.size)) * 100
    
    @property
    def value(self) -> float:
        """Current market value of the position."""
        return self.size * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Original cost basis of the position."""
        return self.size * self.entry_price
    
    def is_stop_triggered(self) -> bool:
        """Check if stop loss or take profit is triggered."""
        if self.direction == TradeDirection.LONG:
            if self.stop_loss and self.current_price <= self.stop_loss:
                return True
            if self.take_profit and self.current_price >= self.take_profit:
                return True
        else:  # SHORT
            if self.stop_loss and self.current_price >= self.stop_loss:
                return True
            if self.take_profit and self.current_price <= self.take_profit:
                return True
        return False

@dataclass
class Trade:
    """
    Represents a completed trade with execution details.
    
    Attributes:
        symbol: Trading symbol
        direction: TradeDirection.LONG or TradeDirection.SHORT
        entry_time: When the position was opened
        exit_time: When the position was closed
        entry_price: Entry price
        exit_price: Exit price
        size: Position size in base currency
        pnl: Realized profit/loss
        pnl_pct: Realized profit/loss as percentage
        commission: Total commissions paid
        slippage: Total slippage cost
        metadata: Additional trade metadata
    """
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> timedelta:
        """Duration the position was held."""
        return self.exit_time - self.entry_time
    
    @property
    def is_winning_trade(self) -> bool:
        """Whether the trade was profitable."""
        return self.pnl > 0

@dataclass
class PortfolioConfig:
    """
    Configuration for the portfolio simulator.
    
    Attributes:
        initial_capital: Starting capital in base currency
        base_currency: Base currency for the portfolio (e.g., 'USD', 'EUR')
        fee_rate: Trading fee as a fraction (e.g., 0.001 for 0.1%)
        slippage: Expected slippage as a fraction (e.g., 0.0005 for 0.05%)
        max_position_size: Maximum position size as fraction of portfolio (0-1)
        max_leverage: Maximum allowed leverage (1.0 = no leverage)
        stop_loss_pct: Default stop loss as percentage (None to disable)
        take_profit_pct: Default take profit as percentage (None to disable)
        max_drawdown_pct: Maximum allowed drawdown before forced liquidation (0-1)
        risk_per_trade: Maximum risk per trade as fraction of portfolio (0-1)
    """
    initial_capital: float = 10000.0
    base_currency: str = 'USD'
    fee_rate: float = 0.001  # 0.1% fee per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.1  # Max 10% of portfolio per position
    max_leverage: float = 1.0  # 1.0 = no leverage
    stop_loss_pct: Optional[float] = 0.05  # 5% stop loss
    take_profit_pct: Optional[float] = 0.10  # 10% take profit
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    risk_per_trade: float = 0.01  # 1% risk per trade

class PortfolioSimulator:
    """
    Advanced portfolio simulator for backtesting trading strategies.
    
    Features:
    - Multi-asset and multi-currency support
    - Realistic trade execution with fees and slippage
    - Position sizing and risk management
    - Performance tracking and reporting
    - Support for long/short positions
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize the portfolio simulator.
        
        Args:
            config: Portfolio configuration (uses defaults if None)
        """
        self.config = config or PortfolioConfig()
        
        # Portfolio state
        self.initial_capital = self.config.initial_capital
        self.available_cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.trade_history: List[Trade] = []
        self.equity_curve = []
        self.timestamps = []
        self.current_time: Optional[datetime] = None
        
        # Performance metrics
        self.realized_pnl = 0.0
        self.total_commissions = 0.0
        self.total_slippage = 0.0
        self.max_drawdown = 0.0
        self.high_water_mark = self.initial_capital
        
        logger.info(f"PortfolioSimulator initialized with {self.config.initial_capital:,.2f} {self.config.base_currency}")
        
    def update_time(self, timestamp: datetime) -> None:
        """
        Update the current simulation time.
        
        Args:
            timestamp: Current timestamp
        """
        self.current_time = timestamp
        self.timestamps.append(timestamp)
        self._update_equity_curve()
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update market prices for all symbols.
        
        Args:
            prices: Dictionary of symbol -> price
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
        
        # Check for stop loss/take profit triggers
        self._check_stop_conditions()
        
        # Update equity curve
        self._update_equity_curve()
    
    def _update_equity_curve(self) -> None:
        """Update the equity curve with current portfolio value."""
        current_value = self.total_value
        self.equity_curve.append(current_value)
        self.high_water_mark = max(self.high_water_mark, current_value)
        
        # Update max drawdown
        if self.high_water_mark > 0:
            self.max_drawdown = max(
                self.max_drawdown,
                (self.high_water_mark - current_value) / self.high_water_mark
            )
    
    def _check_stop_conditions(self) -> None:
        """Check and execute stop loss/take profit orders."""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            if position.is_stop_triggered():
                self.close_position(symbol, position.current_price)
    
    def _calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float],
        risk_fraction: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price (optional)
            risk_fraction: Fraction of portfolio to risk (overrides config if provided)
            
        Returns:
            Position size in base currency
        """
        if risk_fraction is None:
            risk_fraction = self.config.risk_per_trade
            
        # Calculate risk amount in base currency
        risk_amount = self.total_value * risk_fraction
        
        # If no stop loss, use a default position size based on max position size
        if stop_loss is None:
            return self.total_value * self.config.max_position_size
            
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0.0
            
        # Calculate position size with slippage and fees
        position_size = risk_amount / risk_per_unit
        
        # Apply position size limits
        max_position_value = self.total_value * self.config.max_position_size
        max_position_size = max_position_value / entry_price if entry_price > 0 else 0
        
        return min(position_size, max_position_size)
    
    def open_position(
        self,
        symbol: str,
        direction: TradeDirection,
        price: float,
        size: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            direction: TradeDirection.LONG or TradeDirection.SHORT
            price: Entry price
            size: Position size in base currency (calculated if None)
            stop_loss: Stop loss price (uses config if None)
            take_profit: Take profit price (uses config if None)
            metadata: Additional position metadata
            
        Returns:
            True if position was opened successfully, False otherwise
        """
        if not self.current_time:
            logger.error("Cannot open position: current time not set")
            return False
            
        if symbol in self.positions:
            logger.warning(f"Position for {symbol} already exists")
            return False
            
        # Set default stop loss/take profit if not provided
        if stop_loss is None and self.config.stop_loss_pct is not None:
            if direction == TradeDirection.LONG:
                stop_loss = price * (1 - self.config.stop_loss_pct)
            else:  # SHORT
                stop_loss = price * (1 + self.config.stop_loss_pct)
                
        if take_profit is None and self.config.take_profit_pct is not None:
            if direction == TradeDirection.LONG:
                take_profit = price * (1 + self.config.take_profit_pct)
            else:  # SHORT
                take_profit = price * (1 - self.config.take_profit_pct)
        
        # Calculate position size if not provided
        if size is None:
            size = self._calculate_position_size(symbol, price, stop_loss)
            
        # Check if we have enough capital
        required_capital = size * price * (1 + self.config.fee_rate)
        if required_capital > self.available_cash:
            logger.warning(f"Insufficient capital to open position: {required_capital:.2f} > {self.available_cash:.2f}")
            return False
            
        # Apply slippage
        if direction == TradeDirection.LONG:
            execution_price = price * (1 + self.config.slippage)
        else:  # SHORT
            execution_price = price * (1 - self.config.slippage)
            
        # Calculate commission
        commission = size * execution_price * self.config.fee_rate
        
        # Create and store position
        position = Position(
            symbol=symbol,
            entry_price=execution_price,
            size=size,
            direction=direction,
            entry_time=self.current_time,
            current_price=execution_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )
        
        # Update portfolio state
        self.positions[symbol] = position
        self.available_cash -= (size * execution_price + commission)
        self.total_commissions += commission
        self.total_slippage += abs(execution_price - price) * size
        
        logger.info(f"Opened {'LONG' if direction == TradeDirection.LONG else 'SHORT'} position: "
                   f"{symbol} size={size:.4f} @ {execution_price:.2f}")
        
        return True
    
    def close_position(
        self,
        symbol: str,
        price: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            metadata: Additional trade metadata
            
        Returns:
            True if position was closed successfully, False otherwise
        """
        if not self.current_time:
            logger.error("Cannot close position: current time not set")
            return False
            
        if symbol not in self.positions:
            logger.warning(f"No open position found for {symbol}")
            return False
            
        position = self.positions[symbol]
        
        # Apply slippage
        if position.direction == TradeDirection.LONG:
            execution_price = price * (1 - self.config.slippage)
        else:  # SHORT
            execution_price = price * (1 + self.config.slippage)
            
        # Calculate P&L
        if position.direction == TradeDirection.LONG:
            pnl = (execution_price - position.entry_price) * position.size
        else:  # SHORT
            pnl = (position.entry_price - execution_price) * position.size
            
        pnl_pct = (pnl / (position.entry_price * position.size)) * 100 if position.entry_price > 0 else 0
        
        # Calculate commission
        commission = position.size * execution_price * self.config.fee_rate
        
        # Update portfolio state
        self.available_cash += (position.size * execution_price - commission + pnl)
        self.realized_pnl += pnl
        self.total_commissions += commission
        self.total_slippage += abs(execution_price - price) * position.size
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            entry_price=position.entry_price,
            exit_price=execution_price,
            size=position.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=abs(execution_price - price) * position.size,
            metadata=metadata or {}
        )
        self.trade_history.append(trade)
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(f"Closed {position.direction.name} position: {symbol} "
                  f"P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
        
        return True
    
    def close_all_positions(self) -> None:
        """Close all open positions at current market prices."""
        for symbol in list(self.positions.keys()):
            if symbol in self.positions:
                self.close_position(symbol, self.positions[symbol].current_price)
    
    # Performance metrics and reporting
    
    @property
    def leverage(self) -> float:
        """Current portfolio leverage (total exposure / equity)."""
        if self.total_value <= 0:
            return 0.0
        return self.positions_value / self.total_value
    
    @property
    def drawdown(self) -> float:
        """Current drawdown from peak as a percentage."""
        if self.high_water_mark <= 0:
            return 0.0
        return max(0, (self.high_water_mark - self.total_value) / self.high_water_mark)
    
    @property
    def exposure(self) -> Dict[str, float]:
        """Current exposure by asset class."""
        return {
            'total': self.positions_value,
            'long': sum(pos.value for pos in self.positions.values() 
                       if pos.direction == TradeDirection.LONG),
            'short': abs(sum(pos.value for pos in self.positions.values() 
                           if pos.direction == TradeDirection.SHORT)),
            'net': sum(pos.value if pos.direction == TradeDirection.LONG else -pos.value 
                      for pos in self.positions.values())
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return key performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.trade_history:
            return {}
            
        # Basic metrics
        total_return = (self.total_value / self.initial_capital - 1) * 100
        annualized_return = self._calculate_annualized_return()
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        max_drawdown = self.max_drawdown * 100  # Convert to percentage
        
        # Trade statistics
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trade_history) * 100 if self.trade_history else 0
        
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        profit_factor = -np.sum([t.pnl for t in winning_trades]) / \
                       np.sum([abs(t.pnl) for t in losing_trades]) if losing_trades else float('inf')
        
        # Position holding times
        hold_times = [(t.exit_time - t.entry_time).total_seconds() / 86400 for t in self.trade_history]
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        return {
            'portfolio_value': self.total_value,
            'cash': self.available_cash,
            'positions_value': self.positions_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(self.trade_history),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'avg_hold_time_days': avg_hold_time,
            'total_commissions': self.total_commissions,
            'total_slippage': self.total_slippage,
            'leverage': self.leverage,
            'current_drawdown_pct': self.drawdown * 100,
            'exposure': self.exposure
        }
    
    def _calculate_annualized_return(self) -> float:
        """Calculate annualized return."""
        if not self.timestamps or len(self.timestamps) < 2:
            return 0.0
            
        total_days = (self.timestamps[-1] - self.timestamps[0]).days
        if total_days <= 0:
            return 0.0
            
        total_return = self.total_value / self.initial_capital
        annualized_return = (total_return ** (365 / total_days) - 1) * 100
        return annualized_return
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(self.equity_curve) < 2:
            return 0.0
            
        # Calculate daily returns
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # Skip if not enough data
        if len(returns) < 2:
            return 0.0
            
        # Calculate annualized metrics
        annual_factor = np.sqrt(252)  # Assuming daily data
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = np.mean(excess_returns) / np.std(returns, ddof=1) * annual_factor
        
        return float(sharpe)
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Annualized Sortino ratio
        """
        if len(self.equity_curve) < 2:
            return 0.0
            
        # Calculate daily returns
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # Skip if not enough data
        if len(returns) < 2:
            return 0.0
            
        # Calculate downside deviation
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = np.where(returns < 0, returns, 0)
        downside_std = np.std(downside_returns, ddof=1)
        
        # Avoid division by zero
        if downside_std == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
            
        # Annualize
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        return float(sortino)
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as a pandas DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()
            
        data = []
        for trade in self.trade_history:
            data.append({
                'symbol': trade.symbol,
                'direction': trade.direction.name,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'duration_days': trade.duration.total_seconds() / 86400,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'is_win': trade.pnl > 0,
                **trade.metadata
            })
            
        return pd.DataFrame(data)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as a pandas DataFrame."""
        if not self.timestamps or not self.equity_curve:
            return pd.DataFrame()
            
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve,
            'drawdown': self._calculate_drawdown_series()
        })
    
    def _calculate_drawdown_series(self) -> List[float]:
        """Calculate drawdown series."""
        if not self.equity_curve:
            return []
            
        peak = self.equity_curve[0]
        drawdowns = []
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            drawdowns.append(drawdown * 100)  # Convert to percentage
            
        return drawdowns
    
    @property
    def positions_value(self) -> float:
        """Calculate total value of all positions"""
        return sum(pos.quantity * pos.current_price for pos in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value"""
        return self.available_capital + self.positions_value
    
    @property
    def total_pnl(self) -> float:
        """Calculate total PnL (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage"""
        return (self.total_value - self.initial_capital) / self.initial_capital
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current market prices"""
        self.current_prices.update(prices)
        
        # Update position prices and unrealized PnL
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
    
    def execute_trade(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        price: float,
        timestamp: datetime,
        order_type: str = 'market'
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a trade with realistic costs and constraints
        """
        self.current_time = timestamp
        
        try:
            # Validate trade
            validation_result = self._validate_trade(symbol, side, quantity, price)
            if not validation_result['valid']:
                logger.warning(f"Trade validation failed: {validation_result['reason']}")
                return None
            
            # Calculate execution price with slippage
            execution_price = self._calculate_execution_price(price, side, quantity)
            
            # Calculate commission
            trade_value = quantity * execution_price
            commission = trade_value * self.commission_rate
            
            # Execute the trade
            if side.lower() == 'buy':
                result = self._execute_buy(symbol, quantity, execution_price, commission, timestamp)
            else:
                result = self._execute_sell(symbol, quantity, execution_price, commission, timestamp)
            
            if result and result['success']:
                # Record trade
                self._record_trade(symbol, side, quantity, execution_price, commission, timestamp)
                
                return {
                    'success': True,
                    'executed_price': execution_price,
                    'commission': commission,
                    'slippage': abs(execution_price - price),
                    'timestamp': timestamp
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    def _validate_trade(self, symbol: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """Validate trade parameters and constraints"""
        
        # Basic parameter validation
        if quantity <= 0:
            return {'valid': False, 'reason': 'Invalid quantity'}
        
        if price <= 0:
            return {'valid': False, 'reason': 'Invalid price'}
        
        if side.lower() not in ['buy', 'sell']:
            return {'valid': False, 'reason': 'Invalid side'}
        
        # Position limits
        if side.lower() == 'buy' and len(self.positions) >= self.max_positions:
            return {'valid': False, 'reason': 'Maximum positions reached'}
        
        # Capital requirements for buy orders
        if side.lower() == 'buy':
            required_capital = quantity * price * (1 + self.commission_rate + self.slippage_rate)
            required_capital *= self.margin_requirement
            
            if required_capital > self.available_capital:
                return {'valid': False, 'reason': 'Insufficient capital'}
        
        # Position requirements for sell orders
        if side.lower() == 'sell':
            if symbol not in self.positions:
                return {'valid': False, 'reason': 'No position to sell'}
            
            if quantity > self.positions[symbol].quantity:
                return {'valid': False, 'reason': 'Insufficient position size'}
        
        return {'valid': True, 'reason': ''}
    
    def _calculate_execution_price(self, price: float, side: str, quantity: float) -> float:
        """Calculate execution price with slippage"""
        
        # Market impact based on quantity (simplified model)
        market_impact = min(quantity / 10000, 0.01)  # Max 1% impact
        
        # Total slippage
        total_slippage = self.slippage_rate + market_impact
        
        if side.lower() == 'buy':
            # Buy orders execute at higher price
            execution_price = price * (1 + total_slippage)
        else:
            # Sell orders execute at lower price
            execution_price = price * (1 - total_slippage)
        
        return execution_price
    
    def _execute_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Execute buy order"""
        
        total_cost = quantity * price + commission
        
        # Check available capital
        if total_cost > self.available_capital:
            return {'success': False, 'reason': 'Insufficient capital'}
        
        # Update capital
        self.available_capital -= total_cost
        
        # Add or update position
        if symbol in self.positions:
            # Average down existing position
            existing_pos = self.positions[symbol]
            total_quantity = existing_pos.quantity + quantity
            weighted_price = (
                (existing_pos.quantity * existing_pos.entry_price + quantity * price) / 
                total_quantity
            )
            
            existing_pos.quantity = total_quantity
            existing_pos.entry_price = weighted_price
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
                current_price=price
            )
        
        return {'success': True}
    
    def _execute_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Execute sell order"""
        
        if symbol not in self.positions:
            return {'success': False, 'reason': 'No position to sell'}
        
        position = self.positions[symbol]
        
        if quantity > position.quantity:
            return {'success': False, 'reason': 'Insufficient position size'}
        
        # Calculate proceeds
        proceeds = quantity * price - commission
        
        # Calculate realized PnL
        cost_basis = quantity * position.entry_price
        realized_pnl = proceeds - cost_basis
        self.realized_pnl += realized_pnl
        
        # Update capital
        self.available_capital += proceeds
        
        # Update or remove position
        if quantity == position.quantity:
            # Close entire position
            del self.positions[symbol]
        else:
            # Partial close
            position.quantity -= quantity
        
        return {'success': True, 'realized_pnl': realized_pnl}
    
    def _record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float,
        timestamp: datetime
    ) -> None:
        """Record trade in history"""
        
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'portfolio_value': self.total_value
        }
        
        self.trade_history.append(trade_record)
        
        logger.debug(f"Trade recorded: {side} {quantity} {symbol} @ ${price:.4f}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        return {
            'timestamp': self.current_time,
            'total_value': self.total_value,
            'available_capital': self.available_capital,
            'positions_value': self.positions_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'total_return': self.total_return,
            'num_positions': len(self.positions),
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'entry_time': pos.entry_time
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get specific position"""
        return self.positions.get(symbol)
    
    def close_position(self, symbol: str, timestamp: datetime) -> bool:
        """Close entire position"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        current_price = self.current_prices.get(symbol, position.current_price)
        
        return self.execute_trade(
            symbol=symbol,
            side='sell',
            quantity=position.quantity,
            price=current_price,
            timestamp=timestamp
        ) is not None
    
    def close_all_positions(self, timestamp: datetime) -> int:
        """Close all positions"""
        closed_count = 0
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            if self.close_position(symbol, timestamp):
                closed_count += 1
        
        return closed_count
    
    def reset(self) -> None:
        """Reset portfolio to initial state"""
        self.available_capital = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        self.realized_pnl = 0.0
        self.current_prices.clear()
        self.current_time = None
        
        logger.info("Portfolio reset to initial state")
