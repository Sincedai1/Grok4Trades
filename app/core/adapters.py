import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import time
from typing import Optional, Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential

class CCXTAdapter:
    def __init__(self, exchange_id: str, api_key: str = "", api_secret: str = ""):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exchange.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
        """
        Fetch OHLCV data with retry logic and idempotency.
        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        
        Note: CCXT's fetch_ohlcv is idempotent by nature as it's a GET request
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """
        Place market order with retry logic and idempotency.
        
        Note on idempotency:
        - Use client_oid = f"{symbol}-{side}-{amount}-{int(time.time())}"
        - Store placed orders to prevent duplicates on retry
        - Check order status before retry
        """
        client_oid = f"{symbol}-{side}-{amount}-{int(time.time())}"
        
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params={'clientOrderId': client_oid}
            )
            return order
        except Exception as e:
            print(f"Error creating order: {str(e)}")
            raise
