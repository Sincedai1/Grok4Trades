import logging
import asyncio
from pathlib import Path
from typing import Dict, List

import ccxt.async_support as ccxt
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.models import Trade

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.exchange = None
        self.db = SessionLocal()
        self.setup_logging()
        
    def setup_logging(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "trading.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    async def initialize_exchange(self):
        self.exchange = getattr(ccxt, settings.EXCHANGE_NAME.lower())({
            'apiKey': settings.API_KEY,
            'secret': settings.API_SECRET,
            'enableRateLimit': True,
        })
        
    async def fetch_market_data(self, symbol: str) -> Dict:
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return {}
            
    async def execute_trade(self, symbol: str, side: str, amount: float) -> Dict:
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            
            # Log trade to database
            trade = Trade(
                symbol=symbol,
                side=side,
                amount=amount,
                price=order['price'],
                order_id=order['id']
            )
            self.db.add(trade)
            self.db.commit()
            
            return order
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {}
            
    async def run(self):
        await self.initialize_exchange()
        while True:
            try:
                # Main trading logic here
                for symbol in settings.TRADING_PAIRS:
                    market_data = await self.fetch_market_data(symbol)
                    # Implement your trading strategy here
                    
                await asyncio.sleep(settings.UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)
                
    async def cleanup(self):
        if self.exchange:
            await self.exchange.close()
        self.db.close()

def main():
    bot = TradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        asyncio.run(bot.cleanup())
        logger.info("Bot stopped gracefully")

if __name__ == "__main__":
    main()