import asyncio
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from loguru import logger
import numpy as np

@dataclass
class TradeSignal:
    """Data class for trade signals"""
    symbol: str
    action: str  # 'buy' or 'sell'
    price: float
    size: float
    timestamp: float
    reason: str = ""
    confidence: float = 0.0

class MinimalTradingBot:
    """
    Optimized trading bot implementation with performance improvements
    """
    
    def __init__(self, exchange, symbol: str = 'BTC/USDT', paper_trading: bool = True):
        self.exchange = exchange
        self.symbol = symbol
        self.paper_trading = paper_trading
        self.running = False
        self.last_trade_time = 0
        self.positions: Dict[str, float] = {}
        
        # Cached values
        self._last_ticker = None
        self._last_ticker_time = 0
        self._ticker_cache_ttl = 1.0  # 1 second cache
        
        # Hardcoded safety limits (non-negotiable)
        self.MAX_POSITION_SIZE = 50.0  # $50 max position size
        self.DAILY_LOSS_LIMIT = -25.0  # $25 max daily loss
        self.MIN_BALANCE = 100.0      # Minimum required balance to trade
        self.TRADE_COOLDOWN = 60      # 60 seconds between trades
        
        # Trading state
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.last_reset = self._get_market_day()
        
        logger.info(f"Initialized MinimalTradingBot for {symbol}")
    
    def _get_market_day(self) -> str:
        """Get current market day in YYYY-MM-DD format"""
        return time.strftime('%Y-%m-%d', time.gmtime())
    
    def _reset_daily_metrics(self):
        """Reset daily metrics at the start of a new trading day"""
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.last_reset = self._get_market_day()
        logger.info("Daily metrics reset")
    
    async def _get_cached_ticker(self) -> Dict[str, Any]:
        """Get ticker with caching to reduce API calls"""
        current_time = time.time()
        if (current_time - self._last_ticker_time) < self._ticker_cache_ttl and self._last_ticker:
            return self._last_ticker
            
        self._last_ticker = await self.exchange.fetch_ticker(self.symbol)
        self._last_ticker_time = current_time
        return self._last_ticker
    
    async def check_daily_limits(self) -> bool:
        """Check if daily limits are being respected"""
        current_day = self._get_market_day()
        if current_day != self.last_reset:
            self._reset_daily_metrics()
        
        if self.daily_pnl < self.DAILY_LOSS_LIMIT:
            logger.error(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
            
        return True
    
    async def get_position_size(self, price: float) -> float:
        """Calculate position size based on risk parameters"""
        if price <= 0:
            logger.warning(f"Invalid price for position size calculation: {price}")
            return 0.0
            
        # For minimal implementation, use fixed position size
        # that respects max position size
        position_value = min(
            self.MAX_POSITION_SIZE,
            self.MAX_POSITION_SIZE * 0.1  # Never risk more than 10% of max per trade
        )
        
        # Calculate position size in base currency
        position_size = position_value / price
        
        logger.debug(f"Position size: {position_size:.8f} {self.symbol.split('/')[0]} (${position_value:.2f} at ${price:.2f})")
        return position_size
    
    async def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute a trade based on the signal"""
        # Check cooldown period
        current_time = time.time()
        if (current_time - self.last_trade_time) < self.TRADE_COOLDOWN:
            logger.warning(f"Trade cooldown active. Skipping trade for {self.symbol}")
            return False
        
        # Check daily limits
        if not await self.check_daily_limits():
            logger.error("Daily limits exceeded. Not executing trade.")
            return False
        
        try:
            # Get current price using cached ticker
            ticker = await self._get_cached_ticker()
            current_price = ticker['last']
            
            # Calculate position size based on risk parameters
            position_size = await self.get_position_size(current_price)
            
            if position_size <= 0:
                logger.warning("Invalid position size. Skipping trade.")
                return False
            
            # Execute trade (simulated for paper trading)
            logger.info(f"Executing {signal.action.upper()} order: {position_size:.8f} {self.symbol} @ ${current_price:.2f}")
            
            # In a real implementation, this would call the exchange API
            trade_result = {
                'id': f"sim_{int(time.time() * 1000)}",
                'symbol': self.symbol,
                'side': signal.action,
                'price': current_price,
                'amount': position_size,
                'cost': position_size * current_price,
                'datetime': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
                'fee': None,
                'status': 'closed'
            }
            
            # Update position
            if signal.action == 'buy':
                self.positions[self.symbol] = self.positions.get(self.symbol, 0) + position_size
            elif signal.action == 'sell':
                self.positions[self.symbol] = self.positions.get(self.symbol, 0) - position_size
            
            # Update trade metrics
            self.last_trade_time = current_time
            self.trade_count += 1
            
            # Log the trade
            logger.info(f"Trade executed: {trade_result}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False
    
    async def run(self):
        """Optimized main trading loop"""
        self.running = True
        logger.info("Starting trading bot...")
        
        # Initialize timing variables
        last_check = time.time()
        check_interval = 10  # seconds between market data checks
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check for new market day
                current_day = self._get_market_day()
                if current_day != self.last_reset:
                    self._reset_daily_metrics()
                
                # Check if we should continue trading
                if not await self.check_daily_limits():
                    logger.error("Daily loss limit reached. Stopping bot.")
                    break
                
                # Only check market data at specified intervals
                if (current_time - last_check) >= check_interval:
                    # Get market data using cached ticker
                    ticker = await self._get_cached_ticker()
                    
                    # Here you would generate signals using your strategy
                    # For now, we'll just log the current price
                    logger.debug(f"{self.symbol} price: ${ticker['last']:.2f}")
                    
                    # Update last check time
                    last_check = current_time
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.running = False
            logger.info("Trading bot stopped")
