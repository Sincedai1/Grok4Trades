"""
FRED (Federal Reserve Economic Data) data source implementation.
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

class FREDSource(DataSource):
    """
    Data source for fetching economic data from FRED (Federal Reserve Economic Data).
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Common FRED series IDs for FX rates
    FX_SERIES = {
        'DEXUSEU': {'base': 'EUR', 'target': 'USD', 'name': 'US/Euro Foreign Exchange Rate'},
        'DEXJPUS': {'base': 'JPY', 'target': 'USD', 'name': 'Japan / U.S. Foreign Exchange Rate'},
        'DEXUSUK': {'base': 'USD', 'target': 'GBP', 'name': 'U.S. / U.K. Foreign Exchange Rate'},
        'DEXCAUS': {'base': 'CAD', 'target': 'USD', 'name': 'Canada / U.S. Foreign Exchange Rate'},
        'DEXCHUS': {'base': 'CNY', 'target': 'USD', 'name': 'China / U.S. Foreign Exchange Rate'},
        'DEXSZUS': {'base': 'CHF', 'target': 'USD', 'name': 'Switzerland / U.S. Foreign Exchange Rate'},
        'DEXUSAL': {'base': 'USD', 'target': 'AUD', 'name': 'U.S. / Australia Foreign Exchange Rate'},
        'DEXUSNZ': {'base': 'USD', 'target': 'NZD', 'name': 'U.S. / New Zealand Foreign Exchange Rate'},
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize the FRED data source.
        
        Args:
            api_key: FRED API key (default: from FRED_API_KEY env var)
        """
        super().__init__(name="fred")
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        
        if not self.api_key:
            logger.warning("No FRED API key provided. Some endpoints may be rate-limited.")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Make an API request to FRED.
        
        Args:
            endpoint: API endpoint (e.g., 'series/observations')
            params: Request parameters
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Add API key and file type to params
        params = params or {}
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error_code' in data and data['error_code'] is not None:
                raise ValueError(f"FRED API error {data['error_code']}: {data.get('error_message', 'Unknown error')}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        currency: str = 'USD',
        interval: str = '1d',
        start_date: str = None,
        end_date: str = None,
        limit: int = 2000,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from FRED.
        
        Note: FRED doesn't provide OHLCV data directly. This is a placeholder that returns
        daily close prices for the given symbol.
        
        Args:
            symbol: FRED series ID (e.g., 'DEXUSEU' for EUR/USD)
            currency: Not used, kept for API compatibility
            interval: Not used, kept for API compatibility
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of data points to return
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Initialize empty DataFrame with required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        empty_df = pd.DataFrame(columns=required_columns)
        
        try:
            # Get the series data from FRED
            params = {
                'series_id': symbol,
                'limit': min(limit, 10000),  # FRED's max is 10,000
                'sort_order': 'desc',
                'units': 'lin'
            }
            
            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date
                
            data = self._make_request('series/observations', params)
            
            if 'observations' not in data or not data['observations']:
                logger.warning(f"No data returned for FRED series: {symbol}")
                return empty_df
                
            # Convert to DataFrame
            df = pd.DataFrame(data['observations'])
            
            if df.empty:
                return empty_df
                
            # Convert date and value columns
            df['timestamp'] = pd.to_datetime(df['date'])
            df['close'] = pd.to_numeric(df['value'], errors='coerce')
            
            # FRED only provides daily close prices, so we'll use the same value for OHLC
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = 0.0  # FRED doesn't provide volume data
            
            # Add metadata
            df['symbol'] = symbol
            df['currency'] = currency.upper()
            df['interval'] = interval
            df['source'] = 'fred'
            
            # Select and order columns
            result = df[required_columns].copy()
            
            # Sort by timestamp
            result = result.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(result)} data points for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} from FRED: {str(e)}")
            return empty_df
    
    def fetch_fx_rates(
        self,
        base_currency: str = 'EUR',
        target_currency: str = 'USD',
        start_date: str = None,
        end_date: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch foreign exchange rates from FRED.
        
        Args:
            base_currency: Base currency code (e.g., 'EUR')
            target_currency: Target currency code (e.g., 'USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with FX rates
        """
        # Check if we have a direct series for this pair
        series_id = None
        invert_rate = False
        
        # First, check for direct match
        for sid, info in self.FX_SERIES.items():
            if info['base'] == base_currency and info['target'] == target_currency:
                series_id = sid
                invert_rate = False
                break
            elif info['base'] == target_currency and info['target'] == base_currency:
                series_id = sid
                invert_rate = True
                break
        
        # If no direct match, try to find a path through USD
        if series_id is None and target_currency == 'USD':
            # Look for base_currency/USD
            for sid, info in self.FX_SERIES.items():
                if info['base'] == base_currency and info['target'] == 'USD':
                    series_id = sid
                    invert_rate = False
                    break
        
        if series_id is None:
            raise ValueError(f"No FRED series found for {base_currency}/{target_currency} pair")
        
        # Calculate date range
        end_date = end_date or datetime.utcnow().strftime('%Y-%m-%d')
        start_date = start_date or (datetime.utcnow() - timedelta(days=10*365)).strftime('%Y-%m-%d')
        
        # Make the API request
        params = {
            'series_id': series_id,
            'observation_start': start_date,
            'observation_end': end_date,
            'sort_order': 'asc'
        }
        
        data = self._make_request('series/observations', params)
        
        if 'observations' not in data or not data['observations']:
            logger.warning(f"No data returned for FRED series {series_id}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['observations'])
        
        # Clean and format data
        df = df[['date', 'value']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        
        # Invert rate if needed
        if invert_rate:
            df['value'] = 1 / df['value']
        
        # Rename columns
        df = df.rename(columns={
            'date': 'timestamp',
            'value': 'fx_rate'
        })
        
        # Add metadata
        df['base_currency'] = base_currency.upper()
        df['target_currency'] = target_currency.upper()
        df['source'] = 'fred'
        df['series_id'] = series_id
        
        return df
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get information about a FRED series.
        
        Args:
            series_id: FRED series ID (e.g., 'DEXUSEU')
            
        Returns:
            Dictionary with series information
        """
        data = self._make_request(f'series', {'series_id': series_id})
        
        if 'seriess' not in data or not data['seriess']:
            logger.warning(f"No information found for FRED series {series_id}")
            return {}
        
        return data['seriess'][0]
    
    def search_series(self, search_text: str, **kwargs) -> pd.DataFrame:
        """
        Search for FRED series by text.
        
        Args:
            search_text: Text to search for in series titles and descriptions
            **kwargs: Additional search parameters (e.g., limit, offset, etc.)
            
        Returns:
            DataFrame with matching series
        """
        params = {
            'search_text': search_text,
            'limit': kwargs.get('limit', 1000),
            'offset': kwargs.get('offset', 0),
            'sort_order': 'popularity',
            'filter_variable': 'frequency',
            'filter_value': 'Daily'
        }
        
        data = self._make_request('series/search', params)
        
        if 'seriess' not in data or not data['seriess']:
            logger.warning(f"No series found matching: {search_text}")
            return pd.DataFrame()
        
        return pd.DataFrame(data['seriess'])
    
    def get_series_categories(self, series_id: str) -> pd.DataFrame:
        """
        Get the categories for a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            DataFrame with category information
        """
        data = self._make_request('series/categories', {'series_id': series_id})
        
        if 'categories' not in data or not data['categories']:
            logger.warning(f"No categories found for FRED series {series_id}")
            return pd.DataFrame()
        
        return pd.DataFrame(data['categories'])
    
    def get_category_series(self, category_id: int, **kwargs) -> pd.DataFrame:
        """
        Get all series in a FRED category.
        
        Args:
            category_id: FRED category ID
            **kwargs: Additional parameters (e.g., limit, offset, etc.)
            
        Returns:
            DataFrame with series in the category
        """
        params = {
            'category_id': category_id,
            **kwargs
        }
        
        data = self._make_request('category/series', params)
        
        if 'seriess' not in data or not data['seriess']:
            logger.warning(f"No series found in category {category_id}")
            return pd.DataFrame()
            
        return pd.DataFrame(data['seriess'])
        
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
        Fetch OHLCV data from FRED. For FX pairs, returns daily rates.
        For other symbols, tries to find a matching economic indicator.
        
        Args:
            symbol: Asset symbol (e.g., 'EUR' for EUR/USD)
            currency: Quote currency (e.g., 'USD')
            interval: Data interval (only '1d' is supported for FRED)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            **kwargs: Additional parameters
                
        Returns:
            DataFrame with OHLCV data
        """
        if interval != '1d':
            logger.warning("FRED only supports daily data. Using '1d' interval.")
                
        # For FX pairs
        if len(symbol) == 3 and len(currency) == 3:
            try:
                # Get FX rates
                fx_rates = self.fetch_fx_rates(
                    base_currency=symbol,
                    target_currency=currency,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                    
                if fx_rates.empty:
                    return pd.DataFrame()
                        
                # Convert to OHLCV format (using the same value for OHLC since we only have one price per day)
                ohlcv = pd.DataFrame({
                    'timestamp': pd.to_datetime(fx_rates['timestamp']),
                    'open': fx_rates['fx_rate'],
                    'high': fx_rates['fx_rate'],
                    'low': fx_rates['fx_rate'],
                    'close': fx_rates['fx_rate'],
                    'volume': 0.0  # Volume not available from FRED
                })
                    
                return ohlcv.set_index('timestamp')
                    
            except Exception as e:
                logger.error(f"Error fetching FX rates for {symbol}/{currency}: {e}")
                return pd.DataFrame()
                    
        # For economic indicators
        else:
            logger.warning(f"OHLCV data not directly available for {symbol} from FRED. "
                         f"Consider using fetch_fx_rates for currency pairs.")
            return pd.DataFrame()
