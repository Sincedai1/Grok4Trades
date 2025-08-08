#!/usr/bin/env python3
"""
Minimal Trading Bot Runner

This script initializes and runs the Grok4Trades Minimal trading bot with:
- Paper trading mode (simulated exchange)
- Simple Moving Average strategy
- Basic risk management
- File-based logging
- Streamlit dashboard
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
import os
import subprocess
from typing import Dict, Any

# Configure loguru for consistent logging
from loguru import logger
logger.remove()  # Remove default handler
logger.add(
    'trading_bot.log',
    rotation='1 day',
    retention='7 days',
    level='INFO',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}'
)
logger.add(
    sys.stderr,
    level='INFO',
    format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
)

# Add project root to Python path
PROJECT_ROOT = str(Path(__file__).parent.absolute())
sys.path.insert(0, PROJECT_ROOT)

# Import components
from core.minimal_engine import MinimalTradingBot
from strategies.simple_ma import SimpleMAStrategy
from risk.basic_limits import BasicRiskManager

class MockExchange:
    """Mock exchange for paper trading"""
    def __init__(self, initial_balance: float = 100.0):
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.orders = {}
        self.logger = logger
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        # Simulate price movement around $50,000
        import random
        price = 50000 + random.uniform(-500, 500)
        return {
            'symbol': symbol,
            'last': price,
            'bid': price * 0.9995,  # Slightly lower than last
            'ask': price * 1.0005,  # Slightly higher than last
            'timestamp': asyncio.get_event_loop().time(),
        }
    
    async def create_order(self, symbol: str, side: str, amount: float, price: float) -> Dict[str, Any]:
        order_id = f"mock_order_{int(asyncio.get_event_loop().time() * 1000)}"
        cost = amount * price
        fee = cost * 0.001  # 0.1% fee
        
        # Update balance
        if side.lower() == 'buy':
            if cost + fee > self.balance:
                raise ValueError("Insufficient balance")
            self.balance -= (cost + fee)
            self.positions[symbol] = self.positions.get(symbol, 0.0) + amount
        else:  # sell
            if self.positions.get(symbol, 0) < amount:
                raise ValueError("Insufficient position")
            self.balance += (cost - fee)
            self.positions[symbol] -= amount
            if abs(self.positions[symbol]) < 1e-8:  # Close position if near zero
                del self.positions[symbol]
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side.lower(),
            'amount': amount,
            'price': price,
            'cost': cost,
            'fee': fee,
            'status': 'closed',
            'timestamp': asyncio.get_event_loop().time(),
        }
        
        self.orders[order_id] = order
        self.logger.info(f"Executed {side} order: {amount} {symbol} @ {price}")
        return order

async def run_bot(config: Dict[str, Any]) -> None:
    """Initialize and run the trading bot"""
    logger.info("Starting Grok4Trades Minimal...")
    
    # Create necessary directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Initialize components
    exchange = MockExchange(initial_balance=config['initial_balance'])
    strategy = SimpleMAStrategy(
        fast_window=config['strategy']['fast_window'],
        slow_window=config['strategy']['slow_window'],
        min_confidence=config['strategy']['min_confidence']
    )
    
    risk_manager = BasicRiskManager(
        max_risk_per_trade=config['risk']['max_risk_per_trade'],
        max_daily_loss=config['risk']['max_daily_loss'],
        max_position_size=config['risk']['max_position_size'],
        max_leverage=config['risk']['max_leverage'],
        position_timeout_hours=config['risk']['position_timeout_hours']
    )
    
    # Initialize the trading bot
    bot = MinimalTradingBot(
        exchange=exchange,
        symbol=config['symbol'],
        paper_trading=config['paper_trading']
    )
    
    # Start the bot
    try:
        # Start the Streamlit dashboard in a separate process
        dashboard_process = subprocess.Popen([
            'streamlit', 'run', 
            'ui/essential_dashboard.py',
            '--server.port', str(config['dashboard']['port']),
            '--server.address', config['dashboard']['host']
        ])
        
        logger.info(f"Dashboard started at http://{config['dashboard']['host']}:{config['dashboard']['port']}")
        
        # Run the trading bot
        await bot.run()
        
    except asyncio.CancelledError:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.exception("Error in main loop:")
    finally:
        # Cleanup
        if 'dashboard_process' in locals():
            dashboard_process.terminate()
            dashboard_process.wait()
        
        logger.info("Trading bot stopped")

def main() -> None:
    """Main entry point"""
    # Handle keyboard interrupt
    loop = asyncio.get_event_loop()
    
    # Create task for the main bot
    bot_task = loop.create_task(run_bot(CONFIG))
    
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received...")
        bot_task.cancel()
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_handler)
    
    try:
        loop.run_until_complete(bot_task)
    except asyncio.CancelledError:
        pass
    finally:
        # Cleanup
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        
        # Run remaining tasks
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        loop.close()
        logger.info("Shutdown complete")

# Configuration
CONFIG = {
    'symbol': 'BTC/USDT',
    'paper_trading': True,
    'initial_balance': 100.0,  # Start with $100 as per requirements
    'strategy': {
        'fast_window': 10,  # Short MA window
        'slow_window': 20,  # Long MA window
        'min_confidence': 0.6,
    },
    'risk': {
        'max_risk_per_trade': 0.02,  # 2% risk per trade
        'max_daily_loss': 0.05,      # 5% max daily loss (as a decimal between 0 and 0.1)
        'max_position_size': 0.5,    # 50% of portfolio max position size
        'max_leverage': 1.0,         # No leverage initially
        'position_timeout_hours': 24, # 24h position timeout
    },
    'logging': {
        'log_dir': 'logs',
        'max_log_days': 7,           # Keep logs for 7 days
    },
    'dashboard': {
        'port': 8501,                # Streamlit dashboard port
        'host': '0.0.0.0',           # Bind to all network interfaces
    },
}

if __name__ == "__main__":
    main()
