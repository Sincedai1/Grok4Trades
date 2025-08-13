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

# Import our Telegram notifier
from telegram_notifier import TelegramNotifier, AlertType, send_trade_signal, send_error_alert

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBotWithTelegram:
    """Trading bot with integrated Telegram notifications"""
    
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
        
        # Initialize Telegram notifier
        self.telegram = TelegramNotifier()
        
        # Initialize exchange (paper or live)
        self.exchange = self._init_exchange()
        
        # Trading state
        self.running = False
        self.emergency_stop = False
        self.positions = {}
        self.daily_pnl = 0.0
        self.daily_loss_limit = self.max_capital * 0.05  # 5% daily loss limit
        self.trades_today = []
        self.start_time = datetime.now()
        
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
    
    async def start(self):
        """Start the trading bot with Telegram notifications"""
        async with self.telegram:
            # Send startup notification
            await self._send_system_status("Starting")
            
            try:
                await self.run()
            except Exception as e:
                await send_error_alert(
                    component="Trading Bot",
                    error_message=str(e),
                    action="Bot stopped",
                    impact="High"
                )
                raise
    
    async def run(self):
        """Main trading loop with Telegram notifications"""
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
                    
                    # Send periodic status updates (every 30 minutes)
                    if int(time.time()) % 1800 == 0:
                        await self._send_system_status("Running")
                    
                    # Sleep before next iteration (1 minute)
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                    await send_error_alert(
                        component="Main Loop",
                        error_message=str(e),
                        action="Retrying in 5 seconds",
                        impact="Medium"
                    )
                    await asyncio.sleep(5)  # Short delay before retry
        
        except asyncio.CancelledError:
            logger.info("Trading bot stopped by user")
            await self._send_system_status("Stopped")
        
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
            
            # Send signal notification if generated
            if signal:
                logger.info(f"Generated {signal['action']} signal at {current_price}: {signal['reason']}")
                
                # Calculate target and stop loss
                if signal['action'] == 'buy':
                    target = current_price * 1.02  # 2% profit target
                    stop_loss = current_price * 0.98  # 2% stop loss
                else:
                    target = current_price * 0.98
                    stop_loss = current_price * 1.02
                
                # Send Telegram notification
                await send_trade_signal(
                    symbol=self.symbol,
                    side=signal['action'].upper(),
                    entry=current_price,
                    target=target,
                    stop_loss=stop_loss,
                    strategy="MA Crossover",
                    confidence=int(signal['confidence'] * 5)
                )
                
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
    
    async def _execute_trade(self, signal: Dict, market_data: pd.DataFrame):
        """Execute a trade based on the signal with Telegram notifications"""
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
            else:
                order = await self.exchange.create_market_sell_order(
                    symbol=self.symbol,
                    amount=position_size
                )
            
            if order:
                # Send execution notification
                await self.telegram.send_alert(AlertType.TRADE_EXECUTED, {
                    'symbol': self.symbol,
                    'side': signal['action'].upper(),
                    'amount': position_size,
                    'price': order['price'] if 'price' in order else signal['price'],
                    'notional': position_size * signal['price'],
                    'fee': order.get('fee', {'cost': 0})['cost'],
                    'order_id': order['id'],
                    'timestamp': datetime.now()
                })
                
                # Store trade info
                self.trades_today.append({
                    'id': order['id'],
                    'symbol': self.symbol,
                    'side': signal['action'],
                    'amount': position_size,
                    'price': order['price'] if 'price' in order else signal['price'],
                    'timestamp': datetime.now()
                })
                
                logger.info(f"Executed {signal['action']} order: {order['id']}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            await send_error_alert(
                component="Trade Execution",
                error_message=str(e),
                action="Trade cancelled",
                impact="Medium"
            )
    
    async def _monitor_positions(self):
        """Monitor open positions and send alerts"""
        try:
            # Get open positions
            if hasattr(self.exchange, 'fetch_positions'):
                positions = await self.exchange.fetch_positions()
                
                for position in positions:
                    if position['contracts'] > 0:
                        # Check for stop loss or take profit
                        pnl_pct = (position['percentage'] or 0)
                        
                        # Risk alert if position is losing more than 5%
                        if pnl_pct < -5:
                            await self.telegram.send_alert(AlertType.RISK_ALERT, {
                                'alert_type': 'Position at risk',
                                'symbol': position['symbol'],
                                'pnl': position['unrealizedPnl'],
                                'pnl_pct': pnl_pct,
                                'recommended_action': 'Consider closing position'
                            })
                        
                        # Take profit alert if position is gaining more than 10%
                        elif pnl_pct > 10:
                            await self.telegram.send_alert(AlertType.RISK_ALERT, {
                                'alert_type': 'Take profit opportunity',
                                'symbol': position['symbol'],
                                'pnl': position['unrealizedPnl'],
                                'pnl_pct': pnl_pct,
                                'recommended_action': 'Consider taking profits'
                            })
                            
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
    
    async def _check_emergency_conditions(self):
        """Check for emergency stop conditions and send alerts"""
        try:
            # Check daily loss limit
            if abs(self.daily_pnl) > self.daily_loss_limit:
                self.emergency_stop = True
                await self.telegram.send_alert(AlertType.RISK_ALERT, {
                    'alert_type': 'DAILY LOSS LIMIT EXCEEDED',
                    'drawdown': abs(self.daily_pnl / self.max_capital * 100),
                    'daily_loss': abs(self.daily_pnl),
                    'open_positions': len(self.positions),
                    'recommended_action': 'Trading halted - manual intervention required'
                })
                logger.critical(f"Emergency stop activated - Daily loss: ${abs(self.daily_pnl)}")
            
            # Check system health
            if not await self._check_system_health():
                await send_error_alert(
                    component="System Health",
                    error_message="System health check failed",
                    action="Monitoring closely",
                    impact="Medium"
                )
                
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {str(e)}")
    
    async def _calculate_position_size(self, signal: Dict, market_data: pd.DataFrame) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            balance = await self._get_account_balance()
            
            # Calculate position size (max 2% risk per trade)
            risk_amount = balance * self.max_risk_pct
            current_price = market_data['close'].iloc[-1]
            
            # Simple position sizing
            position_size = risk_amount / current_price
            
            # Apply max position limits
            max_position = balance * 0.1 / current_price  # Max 10% of capital per position
            position_size = min(position_size, max_position)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    async def _get_account_balance(self) -> float:
        """Get account balance"""
        try:
            if self.trading_mode == 'paper':
                # Return simulated balance
                return self.max_capital
            else:
                # Get real balance from exchange
                balance = await self.exchange.fetch_balance()
                return balance['USDT']['free'] if 'USDT' in balance else 0
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return 0
    
    async def _update_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate daily P&L
            # This is simplified - in real implementation, track actual trades
            if hasattr(self.exchange, 'fetch_my_trades'):
                trades = await self.exchange.fetch_my_trades(self.symbol, limit=50)
                # Calculate P&L from trades...
                pass
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    async def _check_system_health(self) -> bool:
        """Check system health"""
        try:
            # Check exchange connectivity
            await self.exchange.fetch_ticker(self.symbol)
            
            # Check Redis connectivity
            self.redis.ping()
            
            return True
        except:
            return False
    
    async def _send_system_status(self, status: str):
        """Send system status update"""
        try:
            balance = await self._get_account_balance()
            uptime = datetime.now() - self.start_time
            
            await self.telegram.send_alert(AlertType.SYSTEM_STATUS, {
                'status': status,
                'uptime': str(uptime).split('.')[0],
                'active_strategies': 1,
                'balance': balance,
                'free_margin': balance * 0.8,  # Simplified
                'used_margin': balance * 0.2,  # Simplified
            })
        except Exception as e:
            logger.error(f"Error sending status: {str(e)}")
    
    async def _shutdown(self):
        """Graceful shutdown with final notifications"""
        logger.info("Shutting down trading bot...")
        
        # Send daily summary
        await self._send_daily_summary()
        
        # Close exchange connection
        await self.exchange.close()
        
        # Send shutdown notification
        await self._send_system_status("Stopped")
    
    async def _send_daily_summary(self):
        """Send daily trading summary"""
        try:
            # Calculate statistics
            total_trades = len(self.trades_today)
            winning_trades = [t for t in self.trades_today if t.get('pnl', 0) > 0]
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            
            await self.telegram.send_alert(AlertType.DAILY_SUMMARY, {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': self.daily_pnl,
                'pnl_pct': (self.daily_pnl / self.max_capital * 100) if self.max_capital > 0 else 0,
                'best_trade': 'N/A',  # Would need to track this
                'best_pnl': 0,
                'worst_trade': 'N/A',
                'worst_pnl': 0,
                'max_drawdown': 5.0,  # Would need to calculate
                'sharpe_ratio': 1.5,  # Would need to calculate
                'top_strategy': 'MA Crossover'
            })
        except Exception as e:
            logger.error(f"Error sending daily summary: {str(e)}")

# Example usage
if __name__ == "__main__":
    async def main():
        bot = TradingBotWithTelegram()
        await bot.start()
    
    asyncio.run(main())
