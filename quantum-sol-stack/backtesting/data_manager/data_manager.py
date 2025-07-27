"""
Core Data Manager for handling data operations in the backtesting system.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages data acquisition, validation, and normalization for backtesting.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the DataManager.
        
        Args:
            base_dir: Base directory for storing data files
        """
        from .sources import CryptoCompareSource, KrakenSource, FREDSource
        
        self.base_dir = base_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'historical'
        )
        
        # Initialize data sources
        self.sources = {
            'cryptocompare': CryptoCompareSource(),
            'kraken': KrakenSource(),
            'fred': FREDSource()
        }
        
        # Cache for FX rates and other frequently accessed data
        self.cache = {
            'fx_rates': {},
            'ohlcv': {}
        }
        
        # Ensure data directories exist
        self._ensure_data_directories()
    
    def _ensure_data_directories(self) -> None:
        """Ensure required data directories exist."""
        os.makedirs(os.path.join(self.base_dir, 'crypto'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'fx'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'research'), exist_ok=True)
    
    def _validate_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate that the DataFrame contains all required OHLCV columns.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol for error messages
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected DataFrame for {symbol}, got {type(df)}")
            return False
            
        if df.empty:
            logger.warning(f"Empty DataFrame returned for {symbol}")
            return False
            
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns for {symbol}: {missing_columns}")
            return False
            
        # Check for NaN values in critical columns
        critical_columns = ['timestamp', 'open', 'high', 'low', 'close']
        for col in critical_columns:
            if df[col].isna().any():
                logger.warning(f"Found NaN values in {col} for {symbol}")
                return False
                
        return True
        
    def _standardize_ohlcv_data(self, df: pd.DataFrame, symbol: str, currency: str) -> pd.DataFrame:
        """
        Standardize OHLCV data format.
        
        Args:
            df: Input DataFrame
            symbol: Symbol name
            currency: Currency
            
        Returns:
            Standardized DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Ensure numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add metadata
        df['symbol'] = symbol.upper()
        df['currency'] = currency.upper()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def fetch_ohlcv(
        self, 
        symbol: str, 
        currency: str = 'USD',
        interval: str = '1d',
        start_date: str = None,
        end_date: str = None,
        source: str = 'cryptocompare',
        force_fetch: bool = False,
        normalize: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            currency: Quote currency (e.g., 'USD', 'EUR')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            source: Data source ('cryptocompare', 'kraken', 'fred')
            force_fetch: If True, force fetch new data even if local cache exists
            normalize: If True, normalize the data after fetching
            
        Returns:
            DataFrame with OHLCV data (timestamp, open, high, low, close, volume)
        """
        # Generate cache key
        cache_key = f"{symbol}_{currency}_{source}_{interval}"
        
        # Check cache first
        if not force_fetch and cache_key in self.cache['ohlcv']:
            cached_data = self.cache['ohlcv'][cache_key].copy()
            if self._validate_ohlcv_data(cached_data, symbol):
                return cached_data
        
        # Generate cache file path
        os.makedirs(os.path.join(self.base_dir, 'crypto'), exist_ok=True)
        cache_file = os.path.join(self.base_dir, 'crypto', f"{cache_key}.parquet")
        
        # Try to load from cache if available and not forcing fetch
        if not force_fetch and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if self._validate_ohlcv_data(df, symbol):
                    self.cache['ohlcv'][cache_key] = df
                    return df
                logger.warning("Cached data validation failed, fetching fresh data")
            except Exception as e:
                logger.warning(f"Error loading cached data: {str(e)}")
                
        # If we get here, we need to fetch fresh data
        logger.info(f"Fetching fresh {symbol}/{currency} {interval} data from {source}")
        
        # Fetch data from the appropriate source
        if source not in self.sources:
            raise ValueError(f"Unsupported data source: {source}")
            
        try:
            df = self.sources[source].fetch_ohlcv(
                symbol=symbol,
                currency=currency,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
            
            # Validate the returned data
            if not self._validate_ohlcv_data(df, symbol):
                raise ValueError(f"Invalid data returned from {source} for {symbol}")
                
            # Standardize the data
            df = self._standardize_ohlcv_data(df, symbol, currency)
            
            # Cache the result
            self.cache['ohlcv'][cache_key] = df
            
            # Save to disk
            try:
                df.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Failed to cache data: {str(e)}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} data from {source}: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached data from {cache_file}")
                self.cache['ohlcv'][cache_key] = df
                return df.copy()
            except Exception as e:
                logger.warning(f"Error loading cached data: {e}")
        
        # Fetch data from the appropriate source
        if source not in self.sources:
            raise ValueError(f"Unsupported data source: {source}")
        
        try:
            df = self.sources[source].fetch_ohlcv(
                symbol=symbol,
                currency=currency,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
            
            # Ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Expected DataFrame from {source} but got {type(df)}")
                return pd.DataFrame()
                
            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame()
                
            # Ensure timestamp is a datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    logger.error(f"Failed to convert timestamp to datetime: {e}")
                    return pd.DataFrame()
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Normalize data if requested
            if normalize and not df.empty:
                df = self.normalize_data(df, interval=interval, **kwargs)
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
        
        # Cache the result
        if not df.empty:
            self.cache['ohlcv'][cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                logger.info(f"Saved data to {cache_file}")
            except Exception as e:
                logger.error(f"Error saving data to cache: {e}")
        
        return df
    
    def fetch_fx_rates(
        self,
        base_currency: str = 'EUR',
        target_currency: str = 'USD',
        start_date: str = None,
        end_date: str = None,
        force_fetch: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch foreign exchange rates.
        
        Args:
            base_currency: Base currency code (e.g., 'EUR')
            target_currency: Target currency code (e.g., 'USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            force_fetch: If True, force fetch new data even if local cache exists
            
        Returns:
            DataFrame with FX rates
        """
        # Generate cache key
        cache_key = f"{base_currency}_{target_currency}"
        
        # Check cache first
        if not force_fetch and cache_key in self.cache['fx_rates']:
            return self.cache['fx_rates'][cache_key].copy()
        
        # Generate cache file path
        cache_file = os.path.join(
            self.base_dir, 
            'fx', 
            f"{cache_key}.parquet"
        )
        
        # Try to load from cache if available and not forcing fetch
        if not force_fetch and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded FX rates from cache: {cache_file}")
                self.cache['fx_rates'][cache_key] = df
                return df.copy()
            except Exception as e:
                logger.warning(f"Error loading cached FX rates: {e}")
        
        # Fetch FX rates from FRED
        df = self.sources['fred'].fetch_fx_rates(
            base_currency=base_currency,
            target_currency=target_currency,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        
        # Cache the result
        if not df.empty:
            self.cache['fx_rates'][cache_key] = df
            
            # Save to cache file
            try:
                df.to_parquet(cache_file, index=False)
                logger.info(f"Saved FX rates to {cache_file}")
            except Exception as e:
                logger.error(f"Error saving FX rates to cache: {e}")
        
        return df
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        interval: str = None,
        fill_method: str = 'ffill',
        outlier_threshold: float = 3.0,
        min_volume: float = 0.0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Normalize and clean OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            interval: Target interval for resampling (None for no resampling)
            fill_method: Method for filling missing values ('ffill', 'bfill', 'linear', etc.)
            outlier_threshold: Z-score threshold for detecting outliers (None to disable)
            min_volume: Minimum volume threshold (rows with lower volume will be dropped)
            
        Returns:
            Normalized DataFrame
        """
        from .utils import normalize_data as _normalize_data
        return _normalize_data(
            df=df,
            interval=interval,
            fill_method=fill_method,
            outlier_threshold=outlier_threshold,
            min_volume=min_volume,
            **kwargs
        )
    
    def validate_data(
        self,
        df: pd.DataFrame,
        min_date: str = None,
        max_date: str = None,
        required_columns: List[str] = None,
        check_gaps: bool = True,
        check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Validate OHLCV data for quality and completeness.
        
        Args:
            df: DataFrame with OHLCV data
            min_date: Minimum expected date (inclusive)
            max_date: Maximum expected date (inclusive)
            required_columns: List of required columns
            check_gaps: If True, check for gaps in the time series
            check_duplicates: If True, check for duplicate timestamps
            
        Returns:
            Dictionary with validation results
        """
        from .utils import validate_data as _validate_data
        return _validate_data(
            df=df,
            min_date=min_date,
            max_date=max_date,
            required_columns=required_columns,
            check_gaps=check_gaps,
            check_duplicates=check_duplicates
        )
    
    def convert_currency(
        self,
        df: pd.DataFrame,
        target_currency: str,
        base_currency_col: str = 'currency',
        price_cols: List[str] = None,
        volume_col: str = 'volume',
        inplace: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Convert prices and volumes in a DataFrame to a target currency.
        
        Args:
            df: DataFrame containing price/volume data
            target_currency: Currency to convert to (e.g., 'USD', 'EUR')
            base_currency_col: Column containing the base currency for each row
            price_cols: List of price columns to convert (default: ['open', 'high', 'low', 'close'])
            volume_col: Volume column to convert (default: 'volume')
            inplace: If True, modify the DataFrame in place
            
        Returns:
            DataFrame with converted prices and volumes
        """
        from .utils import convert_currency as _convert_currency
        return _convert_currency(
            df=df,
            target_currency=target_currency,
            base_currency_col=base_currency_col,
            price_cols=price_cols,
            volume_col=volume_col,
            inplace=inplace,
            data_manager=self,
            **kwargs
        )

# Example usage
if __name__ == "__main__":
    # Initialize data manager
    dm = DataManager()
    
    # Example 1: Fetch BTC/USD daily data from CryptoCompare
    print("Fetching BTC/USD daily data...")
    btc_daily = dm.fetch_ohlcv('BTC', 'USD', '1d')
    print(f"Fetched {len(btc_daily)} rows of BTC/USD daily data")
    
    # Example 2: Fetch BTC/EUR hourly data from Kraken
    print("\nFetching BTC/EUR hourly data...")
    btc_eur = dm.fetch_ohlcv('BTC', 'EUR', '1h', source='kraken')
    print(f"Fetched {len(btc_eur)} rows of BTC/EUR hourly data")
    
    # Example 3: Convert BTC/EUR to BTC/USD
    print("\nConverting BTC/EUR to BTC/USD...")
    btc_usd = dm.convert_currency(btc_eur, 'USD')
    print(f"Converted {len(btc_usd)} rows to USD")
    
    # Example 4: Validate data
    print("\nValidating data...")
    validation = dm.validate_data(btc_usd)
    print(f"Data is {'valid' if validation['is_valid'] else 'invalid'}")
    if not validation['is_valid']:
        print("Issues found:", "\n- ".join([""] + validation['issues']))
