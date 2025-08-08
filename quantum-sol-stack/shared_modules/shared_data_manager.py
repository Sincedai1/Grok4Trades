import redis
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta

class SharedDataManager:
    def __init__(self, redis_host='redis-cache', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
    def cache_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame, ttl: int = 300):
        """Cache market data with 5-minute TTL"""
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            # Convert DataFrame to JSON for Redis storage
            data_json = data.to_json(orient='records')
            self.redis_client.setex(cache_key, ttl, data_json)
            self.logger.info(f"Cached market data for {symbol} {timeframe}")
        except Exception as e:
            self.logger.error(f"Failed to cache market data: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Retrieve cached market data"""
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pd.read_json(cached_data, orient='records')
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached data: {e}")
            return None
    
    def cache_indicators(self, symbol: str, indicators: Dict, ttl: int = 60):
        """Cache calculated technical indicators"""
        try:
            cache_key = f"indicators:{symbol}"
            # Convert numpy arrays to lists for JSON serialization
            serializable_indicators = {}
            for key, value in indicators.items():
                if isinstance(value, np.ndarray):
                    serializable_indicators[key] = value.tolist()
                else:
                    serializable_indicators[key] = value
            
            self.redis_client.setex(cache_key, ttl, json.dumps(serializable_indicators))
        except Exception as e:
            self.logger.error(f"Failed to cache indicators: {e}")
    
    def get_indicators(self, symbol: str) -> Optional[Dict]:
        """Retrieve cached indicators"""
        try:
            cache_key = f"indicators:{symbol}"
            cached_indicators = self.redis_client.get(cache_key)
            if cached_indicators:
                return json.loads(cached_indicators)
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached indicators: {e}")
            return None

# Memory-efficient data streaming
class StreamingDataProcessor:
    def __init__(self, max_buffer_size: int = 1000):
        self.buffers = {}
        self.max_buffer_size = max_buffer_size
        
    def add_tick(self, symbol: str, tick_data: Dict):
        """Add new tick data with automatic buffer management"""
        if symbol not in self.buffers:
            self.buffers[symbol] = []
        
        self.buffers[symbol].append(tick_data)
        
        # Keep only recent data to manage memory
        if len(self.buffers[symbol]) > self.max_buffer_size:
            self.buffers[symbol] = self.buffers[symbol][-self.max_buffer_size:]
    
    def get_recent_data(self, symbol: str, periods: int = 100) -> List[Dict]:
        """Get recent tick data for analysis"""
        if symbol in self.buffers:
            return self.buffers[symbol][-periods:]
        return []
    
    def calculate_streaming_indicators(self, symbol: str) -> Dict:
        """Calculate indicators from streaming data"""
        recent_data = self.get_recent_data(symbol, 50)
        if len(recent_data) < 20:
            return {}
        
        prices = [float(tick['price']) for tick in recent_data]
        
        # Simple moving averages
        sma_20 = sum(prices[-20:]) / 20
        sma_10 = sum(prices[-10:]) / 10
        
        # RSI calculation
        rsi = self._calculate_rsi(prices)
        
        return {
            'sma_20': sma_20,
            'sma_10': sma_10,
            'rsi': rsi,
            'current_price': prices[-1],
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_rsi(self, prices: List[float], periods: int = 14) -> float:
        """Memory-efficient RSI calculation"""
        if len(prices) < periods + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas[-periods:]]
        losses = [-delta if delta < 0 else 0 for delta in deltas[-periods:]]
        
        avg_gain = sum(gains) / periods if gains else 0
        avg_loss = sum(losses) / periods if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
