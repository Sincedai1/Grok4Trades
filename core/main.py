import os
import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import redis
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalTradingBot:
    """Minimal trading bot with essential safety features"""
    
    def __init__(self):
        # Initialize with safe defaults
        self.trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
        self.max_capital = float(os.getenv('MAX_CAPITAL', 100.0))
        self.max_risk_pct = float(os.getenv('MAX_RISK_PCT', 0.02))
        self.symbol = os.getenv('SYMBOL', 'BTC/USDT')
        self.exchange_name = os.getenv('EXCHANGE', 'binance').lower()
        
        # Initialize Redis for state management
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis-cache:6379')
        self.redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
        
        # Initialize exchange (paper or live)
        self.exchange = self._init_exchange()
        
        # Trading state
        self.running = False
        self.emergency_stop = False
        self.positions = {}
        self.daily_pnl = 0.0
        self.daily_loss_limit = self.max_capital * 0.05  # 5% daily loss limit
        
        logger.info(f"Initialized {self.trading_mode.upper()} trading bot with ${self.max_capital} max capital")
    
    def _init_exchange(self):
        """Initialize exchange connection"""
        exchange_class = getattr(ccxt, self.exchange_name)
        
        if self.trading_mode == 'paper':
            # Use testnet for paper trading
            exchange = exchange_class({
                'apiKey': os.getenv('TESTNET_API_KEY', ''),
                'secret': os.getenv('TESTNET_SECRET', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'test': True,  # Use testnet
                },
            })
            logger.info(f"Connected to {self.exchange_name.upper()} testnet")
        else:
            # Live trading with real funds (use with caution)
            exchange = exchange_class({
                'apiKey': os.getenv('LIVE_API_KEY', ''),
                'secret': os.getenv('LIVE_SECRET', ''),
                'enableRateLimit': True,
            })
            logger.warning(f"LIVE TRADING ENABLED - Connected to {self.exchange_name.upper()}")
        
        return exchange
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting trading bot...")
        
        try:
            while self.running and not self.emergency_stop:
                try:
                    # Check for emergency stop conditions
                    await self._check_emergency_conditions()
                    
                    # Get market data
                    market_data = await self._get_market_data()
                    
                    # Generate trading signals
                    signal = await self._generate_signal(market_data)
                    
                    # Execute trades based on signals
                    if signal:
                        await self._execute_trade(signal, market_data)
                    
                    # Monitor open positions
                    await self._monitor_positions()
                    
                    # Update performance metrics
                    await self._update_metrics()
                    
                    # Sleep before next iteration (1 minute)
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                    await asyncio.sleep(5)  # Short delay before retry
        
        except asyncio.CancelledError:
            logger.info("Trading bot stopped by user")
        
        finally:
            await self._shutdown()
    
    async def _get_market_data(self) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            # Get 1-minute candles for the last 100 periods
            ohlcv = await self.exchange.fetch_ohlcv(
                self.symbol, 
                timeframe='1m', 
                limit=100
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise
    
    async def _generate_signal(self, market_data: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signals based on market data"""
        try:
            # Simple moving average strategy
            fast_ma = market_data['close'].rolling(window=5).mean()
            slow_ma = market_data['close'].rolling(window=10).mean()
            
            current_price = market_data['close'].iloc[-1]
            
            # Generate signals
            signal = None
            
            # Bullish signal (fast MA crosses above slow MA)
            if fast_ma.iloc[-2] <= slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
                signal = {
                    'action': 'buy',
                    'price': current_price,
                    'timestamp': datetime.utcnow().isoformat(),
                    'reason': 'MA crossover (5/10) bullish',
                    'confidence': 0.7  # 0-1 confidence score
                }
            # Bearish signal (fast MA crosses below slow MA)
            elif fast_ma.iloc[-2] >= slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
                signal = {
                    'action': 'sell',
                    'price': current_price,
                    'timestamp': datetime.utcnow().isoformat(),
                    'reason': 'MA crossover (5/10) bearish',
                    'confidence': 0.7
                }
            
            # Log the signal
            if signal:
                logger.info(f"Generated {signal['action']} signal at {current_price}: {signal['reason']}")
                
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
    
    async def _execute_trade(self, signal: Dict, market_data: pd.DataFrame):
        """Execute a trade based on the signal"""
        try:
            if self.emergency_stop:
                logger.warning("Trade execution blocked - Emergency stop active")
                return
            
            # Calculate position size based on risk management
            position_size = await self._calculate_position_size(signal, market_data)
            
            if position_size <= 0:
                logger.warning("Position size too small, skipping trade")
                return
            
            # Execute the trade
            order = None
            if signal['action'] == 'buy':
                order = await self.exchange.create_market_buy_order(
                    symbol=self.symbol,
                    amount=position_size
                )
            else:  # sell
                order = await self.exchange.create_market_sell_order(
                    symbol=self.symbol,
                    amount=position_size
                )
            
            # Record the trade
            await self._record_trade(signal, order, position_size)
            
            logger.info(f"Executed {signal['action']} order: {position_size} {self.symbol}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
    
    async def _calculate_position_size(self, signal: Dict, market_data: pd.DataFrame) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get current price and account balance
            current_price = market_data['close'].iloc[-1]
            balance = await self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            
            # Calculate position size based on risk percentage
            risk_amount = min(usdt_balance, self.max_capital) * self.max_risk_pct
            
            # Simple position sizing (1% per trade)
            position_size = (risk_amount / current_price) * 0.01
            
            # Round to appropriate precision
            markets = await self.exchange.load_markets()
            precision = markets[self.symbol]['precision']['amount']
            position_size = round(position_size, precision)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    async def _monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            # Get open positions
            positions = await self.exchange.fetch_positions([self.symbol])
            
            for position in positions:
                if float(position['contracts']) > 0:
                    # Check if position needs to be closed based on stop loss/take profit
                    await self._check_position_exit(position)
        
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
    
    async def _check_position_exit(self, position: Dict):
        """Check if a position should be closed"""
        try:
            # Simple exit strategy: 2% profit or 1% loss
            entry_price = float(position['entryPrice'])
            current_price = float(position['markPrice'] if 'markPrice' in position else position['liquidationPrice'])
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            if pnl_pct >= 2.0 or pnl_pct <= -1.0:
                # Close the position
                side = 'sell' if position['side'] == 'long' else 'buy'
                await self.exchange.create_market_order(
                    symbol=self.symbol,
                    side=side,
                    amount=abs(float(position['contracts'])),
                    params={'reduceOnly': True}
                )
                
                logger.info(f"Closed position: {position['side']} {position['contracts']} {self.symbol} at {current_price} (PnL: {pnl_pct:.2f}%)")
        
        except Exception as e:
            logger.error(f"Error checking position exit: {str(e)}")
    
    async def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        try:
            # Check daily PnL
            if self.daily_pnl <= -self.daily_loss_limit:
                self.emergency_stop = True
                logger.critical(f"EMERGENCY STOP: Daily loss limit reached (-{abs(self.daily_pnl):.2f}%)")
                
                # Close all positions
                await self._close_all_positions()
        
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {str(e)}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            positions = await self.exchange.fetch_positions()
            
            for position in positions:
                if float(position['contracts']) > 0:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    await self.exchange.create_market_order(
                        symbol=position['symbol'],
                        side=side,
                        amount=abs(float(position['contracts'])),
                        params={'reduceOnly': True}
                    )
                    
                    logger.info(f"Closed position (emergency): {position['side']} {position['contracts']} {position['symbol']}")
        
        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")
    
    async def _update_metrics(self):
        """Update performance metrics in Redis"""
        try:
            # Get current balance and PnL
            balance = await self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['total'] if 'USDT' in balance else 0
            
            # Update Redis with current state
            self.redis.hset('trading:status', mapping={
                'running': str(self.running),
                'emergency_stop': str(self.emergency_stop),
                'balance': str(usdt_balance),
                'last_updated': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    async def _record_trade(self, signal: Dict, order: Dict, position_size: float):
        """Record trade details in Redis"""
        try:
            trade_id = order.get('id', str(int(time.time())))
            trade_data = {
                'id': trade_id,
                'symbol': self.symbol,
                'side': signal['action'],
                'price': str(order.get('price', signal['price'])),
                'amount': str(position_size),
                'cost': str(float(position_size) * float(order.get('price', signal['price']))),
                'timestamp': datetime.utcnow().isoformat(),
                'reason': signal.get('reason', ''),
                'confidence': str(signal.get('confidence', 0))
            }
            
            # Store trade in Redis
            self.redis.hset(f'trading:trades:{trade_id}', mapping=trade_data)
            self.redis.zadd('trading:trade_history', {trade_id: time.time()})
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
    
    async def _shutdown(self):
        """Cleanup resources"""
        self.running = False
        logger.info("Shutting down trading bot...")
        
        try:
            # Close exchange connection
            await self.exchange.close()
            logger.info("Exchange connection closed")
            
            # Close Redis connection
            self.redis.close()
            logger.info("Redis connection closed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


async def main():
    """Main entry point"""
    # Initialize and run the trading bot
    bot = MinimalTradingBot()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        await bot._shutdown()


if __name__ == "__main__":
    asyncio.run(main())
