"""
CryptoCompare data source implementation.
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

class CryptoCompareSource(DataSource):
    """
    Data source for fetching cryptocurrency data from CryptoCompare API.
    """
    
    BASE_URL = "https://min-api.cryptocompare.com/data"
    
    def __init__(self, api_key: str = None):
        """
        Initialize the CryptoCompare data source.
        
        Args:
            api_key: CryptoCompare API key (default: from CRYPTOCOMPARE_API_KEY env var)
        """
        super().__init__(name="cryptocompare")
        self.api_key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
        
        # Update session headers
        if hasattr(self, 'session'):
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            })
        
        if not self.api_key:
            logger.warning("No CryptoCompare API key provided. Some endpoints may be rate-limited.")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Make an API request to CryptoCompare.
        
        Args:
            endpoint: API endpoint (e.g., 'histoday')
            params: Request parameters
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Add API key to params if available
        if self.api_key:
            params = params or {}
            params['api_key'] = self.api_key
        
        try:
            # Add API key to headers if available
            headers = {}
            if self.api_key:
                headers['authorization'] = f'Apikey {self.api_key}'
            
            logger.debug(f"Making request to {url} with params: {params}")
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Log raw response for debugging
            raw_data = response.text
            logger.debug(f"Raw response: {raw_data[:500]}...")  # Log first 500 chars
            
            try:
                data = response.json()
                logger.debug(f"Parsed JSON response type: {type(data)}")
                if isinstance(data, dict):
                    logger.debug(f"Response keys: {list(data.keys())}")
                elif isinstance(data, list):
                    logger.debug(f"Response list length: {len(data)}")
                    if data and len(data) > 0:
                        logger.debug(f"First item type: {type(data[0])}")
            except Exception as json_err:
                logger.error(f"Failed to parse JSON response: {json_err}")
                logger.error(f"Response content: {raw_data[:1000]}")
                raise ValueError(f"Invalid JSON response: {str(json_err)}")
            
                error_msg = data.get('Message', 'Unknown error')
                logger.error(f"CryptoCompare API error: {error_msg}")
                return {'Response': 'Error', 'Message': error_msg}
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {'Response': 'Error', 'Message': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {'Response': 'Error', 'Message': str(e)}
    
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
        Fetch OHLCV data from CryptoCompare.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            currency: Quote currency (e.g., 'USD')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of data points to fetch (max 2000)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Initialize empty DataFrame with required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        empty_df = pd.DataFrame(columns=required_columns)
        
        try:
            # Map interval to CryptoCompare's format
            interval_map = {
                '1m': 'histominute',
                '5m': 'histominute',
                '15m': 'histominute',
                '1h': 'histohour',
                '1d': 'histoday'
            }
            
            if interval not in interval_map:
                logger.error(f"Unsupported interval: {interval}")
                return empty_df
                
            endpoint = interval_map[interval]
            
            # Set up date range
            end_date = pd.to_datetime(end_date or datetime.utcnow())
            if start_date:
                start_date = pd.to_datetime(start_date)
            else:
                start_date = end_date - timedelta(days=365)
            
            # Make the API request
            params = {
                'fsym': symbol.upper(),
                'tsym': currency.upper(),
                'limit': min(limit, 2000),  # API max is 2000
                'toTs': int(end_date.timestamp()),
                'e': 'CCCAGG'  # Use CCCAGG as the exchange
            }
            
            logger.info(f"Fetching {symbol}/{currency} {interval} data from {start_date} to {end_date}")
            
            # Make the request
            data = self._make_request(endpoint, params)
            
            # Check for errors
            if not data or 'Data' not in data or not data['Data']:
                logger.error(f"No data returned for {symbol}/{currency}")
                return empty_df
            
            # Convert to DataFrame
            df = pd.DataFrame(data['Data'])
            
            if df.empty:
                logger.warning(f"Empty DataFrame returned for {symbol}/{currency}")
                return empty_df
            
            # Rename and select columns
            column_map = {
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume',
                'volumeto': 'volume_quote'
            }
            
            # Only keep columns that exist in the response
            existing_columns = [col for col in column_map.keys() if col in df.columns]
            df = df[existing_columns].rename(columns=column_map)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in df.columns:
                    if col == 'volume':
                        df['volume'] = 0.0  # Default to 0 if volume is missing
                    else:
                        logger.error(f"Required column '{col}' not found in response")
                        return empty_df
            
            # Filter by date range and sort
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Ensure proper data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add metadata
            df['symbol'] = symbol.upper()
            df['currency'] = currency.upper()
            df['interval'] = interval
            df['source'] = 'cryptocompare'
            
            logger.info(f"Successfully fetched {len(df)} rows of OHLCV data for {symbol}/{currency}")
            return df[required_columns]  # Only return required columns
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}/{currency}: {str(e)}", exc_info=True)
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
        Fetch foreign exchange rates from CryptoCompare.
        
        Args:
            base_currency: Base currency code (e.g., 'EUR')
            target_currency: Target currency code (e.g., 'USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with FX rates
        """
        # For CryptoCompare, we can use the price endpoint with historical data
        endpoint = 'price'
        params = {
            'fsym': base_currency.upper(),
            'tsyms': target_currency.upper()
        }
        
        # Get current rate
        data = self._make_request(endpoint, params)
        
        # Create a DataFrame with the current rate
        df = pd.DataFrame([{
            'timestamp': datetime.utcnow(),
            'base_currency': base_currency.upper(),
            'target_currency': target_currency.upper(),
            'fx_rate': data.get(target_currency.upper(), float('nan')),
            'source': 'cryptocompare'
        }])
        
        # For historical data, we would need to make multiple requests
        # or use a different endpoint, but this is a simplified version
        
        return df
    
    def get_coin_list(self) -> pd.DataFrame:
        """
        Get a list of all available coins on CryptoCompare.
        
        Returns:
            DataFrame with coin information
        """
        endpoint = 'all/coinlist'
        data = self._make_request(endpoint)
        
        if 'Data' not in data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        coins = []
        for coin_id, coin_data in data['Data'].items():
            coin_data['id'] = coin_id
            coins.append(coin_data)
        
        return pd.DataFrame(coins)
    
    def get_exchanges(self) -> pd.DataFrame:
        """
        Get a list of all available exchanges on CryptoCompare.
        
        Returns:
            DataFrame with exchange information
        """
        endpoint = 'data/v4/all/exchanges'
        data = self._make_request(endpoint)
        
        if not isinstance(data, dict):
            return pd.DataFrame()
        
        # Convert to DataFrame
        exchanges = []
        for exchange, coins in data.items():
            for coin, pairs in coins.items():
                for pair in pairs:
                    exchanges.append({
                        'exchange': exchange,
                        'coin': coin,
                        'pair': pair
                    })
        
        return pd.DataFrame(exchanges)
