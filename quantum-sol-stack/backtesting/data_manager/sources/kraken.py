"""
Kraken data source implementation.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

from .base import DataSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class KrakenSource(DataSource):
    """
    Data source for fetching cryptocurrency data from Kraken API.
    """
    
    BASE_URL = "https://api.kraken.com/0/public"
    
    # Mapping of common symbols to Kraken's format
    SYMBOL_MAP = {
        'BTC': 'XBT',  # Kraken uses XBT instead of BTC
        'XBT': 'XBT',
        'ETH': 'ETH',
        'SOL': 'SOL',
        'EUR': 'EUR',
        'USD': 'USD',
        'USDT': 'USDT',
        'USDC': 'USDC',
        'DAI': 'DAI',
    }
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize the Kraken data source.
        
        Args:
            api_key: Kraken API key (optional for public endpoints)
            api_secret: Kraken API secret (optional for public endpoints)
        """
        super().__init__(name="kraken")
        self.api_key = api_key or os.getenv("KRAKEN_API_KEY")
        self.api_secret = api_secret or os.getenv("KRAKEN_API_SECRET")
    
    def _get_pair_name(self, base: str, quote: str) -> str:
        """
        Get the Kraken pair name for a base/quote currency pair.
        
        Args:
            base: Base currency symbol (e.g., 'BTC')
            quote: Quote currency symbol (e.g., 'USD')
            
        Returns:
            Kraken pair name (e.g., 'XBTUSD')
        """
        base = self.SYMBOL_MAP.get(base.upper(), base.upper())
        quote = self.SYMBOL_MAP.get(quote.upper(), quote.upper())
        
        # Special handling for some pairs
        if base == 'XBT' and quote == 'USD':
            return 'XXBTZUSD'
        elif base == 'XBT' and quote == 'EUR':
            return 'XXBTZEUR'
        elif base == 'ETH' and quote == 'USD':
            return 'XETHZUSD'
        elif base == 'ETH' and quote == 'EUR':
            return 'XETHZEUR'
        elif base == 'SOL' and quote == 'USD':
            return 'SOLUSD'
        elif base == 'SOL' and quote == 'EUR':
            return 'SOLEUR'
        else:
            # Try to construct pair name
            return f"X{base}Z{quote}" if len(quote) == 3 else f"X{base}{quote}"
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Make an API request to Kraken.
        
        Args:
            endpoint: API endpoint (e.g., 'OHLC')
            params: Request parameters
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data and data['error']:
                error_msg = data['error'][0] if isinstance(data['error'], list) else str(data['error'])
                raise ValueError(f"Kraken API error: {error_msg}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        currency: str = 'USD',
        interval: str = '1h',
        start_date: str = None,
        end_date: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Kraken.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            currency: Quote currency (e.g., 'USD')
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        # Map interval to Kraken's format (in minutes)
        interval_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        if interval not in interval_map:
            raise ValueError(f"Unsupported interval: {interval}. Must be one of: {', '.join(interval_map.keys())}")
        
        interval_minutes = interval_map[interval]
        
        # Get the Kraken pair name
        pair = self._get_pair_name(symbol, currency)
        
        # Calculate date range
        end_date = pd.to_datetime(end_date or datetime.utcnow())
        if start_date:
            start_date = pd.to_datetime(start_date)
        else:
            # Default to 1 year of data (Kraken has limits on historical data)
            start_date = end_date - timedelta(days=365)
        
        # Convert to timestamps
        since = int(start_date.timestamp())
        to = int(end_date.timestamp())
        
        # Make the API request
        params = {
            'pair': pair,
            'interval': interval_minutes,
            'since': since
        }
        
        data = self._make_request('OHLC', params)
        
        if 'result' not in data or not data['result']:
            logger.warning(f"No data returned for {pair} {interval}")
            return pd.DataFrame()
        
        # Extract the OHLCV data (the first key in result is the pair name)
        pair_key = next(iter(data['result']))
        ohlcv_data = data['result'][pair_key]
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_data,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 
                'volume', 'count'
            ]
        )
        
        if df.empty:
            logger.warning(f"Empty data returned for {pair} {interval}")
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Convert string columns to float
        for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
            df[col] = df[col].astype(float)
        
        # Add metadata
        df['symbol'] = symbol.upper()
        df['currency'] = currency.upper()
        df['interval'] = interval
        df['source'] = 'kraken'
        
        # Filter by date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Standardize the data
        df = self._standardize_ohlcv_data(
            df,
            symbol=symbol.upper(),
            currency=currency.upper(),
            interval=interval,
            source='kraken'
        )
        
        return df
    
    def get_asset_pairs(self) -> pd.DataFrame:
        """
        Get a list of all available asset pairs on Kraken.
        
        Returns:
            DataFrame with asset pair information
        """
        data = self._make_request('AssetPairs')
        
        if 'result' not in data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        pairs = []
        for pair_id, pair_data in data['result'].items():
            pair_data['pair_id'] = pair_id
            pairs.append(pair_data)
        
        return pd.DataFrame(pairs)
    
    def get_assets(self) -> pd.DataFrame:
        """
        Get a list of all available assets on Kraken.
        
        Returns:
            DataFrame with asset information
        """
        data = self._make_request('Assets')
        
        if 'result' not in data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        assets = []
        for asset_id, asset_data in data['result'].items():
            asset_data['asset_id'] = asset_id
            assets.append(asset_data)
        
        return pd.DataFrame(assets)
