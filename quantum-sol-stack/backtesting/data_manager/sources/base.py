"""
Base class for data sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSource(ABC):
    """
    Abstract base class for data sources.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the data source.
        
        Args:
            name: Name of the data source
            **kwargs: Additional source-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._session = None
    
    @property
    def session(self):
        """Lazy-loading session for HTTP requests."""
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
            
            # Create session with retry
            self._session = requests.Session()
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
            
            # Set default headers
            self._session.headers.update({
                'User-Agent': 'QuantumSol/1.0',
                'Accept': 'application/json',
            })
        
        return self._session
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        currency: str = 'USD',
        interval: str = '1d',
        start_date: str = None,
        end_date: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH')
            currency: Quote currency (e.g., 'USD', 'EUR')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            **kwargs: Additional parameters specific to the data source
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    def fetch_fx_rates(
        self,
        base_currency: str,
        target_currency: str,
        start_date: str = None,
        end_date: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch foreign exchange rates.
        
        Args:
            base_currency: Base currency code (e.g., 'EUR')
            target_currency: Target currency code (e.g., 'USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            **kwargs: Additional parameters specific to the data source
            
        Returns:
            DataFrame with FX rates
        """
        raise NotImplementedError("FX rates not supported by this data source")
    
    def _parse_timestamp(self, timestamp: Any, unit: str = 's') -> pd.Timestamp:
        """
        Parse a timestamp into a pandas Timestamp.
        
        Args:
            timestamp: Timestamp to parse (can be int, str, datetime, etc.)
            unit: Time unit if timestamp is numeric ('s' for seconds, 'ms' for milliseconds)
            
        Returns:
            pandas.Timestamp
        """
        return pd.to_datetime(timestamp, unit=unit, utc=True)
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data for required columns and basic sanity checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for missing values
        if df[required_columns].isnull().any().any():
            logger.warning("Found missing values in OHLCV data")
        
        # Check for zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        if (df[price_columns] <= 0).any().any():
            logger.warning("Found zero or negative prices in OHLCV data")
        
        # Check for negative volume
        if (df['volume'] < 0).any():
            logger.warning("Found negative volume in OHLCV data")
        
        return True
    
    def _standardize_ohlcv_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Standardize OHLCV data to a common format.
        
        Args:
            df: Input DataFrame with OHLCV data
            **kwargs: Additional metadata (symbol, currency, interval, etc.)
            
        Returns:
            Standardized DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure timestamp is a datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Standardize column names
        column_mapping = {
            # Timestamp
            'time': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            
            # OHLC
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'price': 'close',
            
            # Volume
            'v': 'volume',
            'vol': 'volume',
            'volumefrom': 'volume',
            'volumeto': 'volume_quote',
            'base_volume': 'volume',
            'quote_volume': 'volume_quote',
            
            # Trades
            'trades': 'trades',
            'count': 'trades',
            'n': 'trades',
            
            # VWAP
            'vwap': 'vwap',
            'average': 'vwap',
        }
        
        # Apply column name mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Add metadata columns
        for key, value in kwargs.items():
            if key not in df.columns:
                df[key] = value
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Reset index to make timestamp a column again
        df = df.reset_index()
        
        return df
    
    def close(self):
        """Clean up resources."""
        if self._session is not None:
            self._session.close()
            self._session = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
