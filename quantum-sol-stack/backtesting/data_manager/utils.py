"""
Utility functions for data processing and manipulation.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(
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
    if df.empty:
        return {
            'is_valid': False,
            'message': 'Empty DataFrame',
            'issues': ['DataFrame is empty']
        }
    
    issues = []
    
    # Check required columns
    if required_columns is None:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for missing values in required columns
    for col in required_columns:
        if col in df.columns and df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            issues.append(f"Column '{col}' has {missing_count} missing values")
    
    # Check date range
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        
        if min_date:
            min_date = pd.to_datetime(min_date)
            if timestamps.min() > min_date:
                issues.append(f"Data starts after min_date: {timestamps.min()} > {min_date}")
        
        if max_date:
            max_date = pd.to_datetime(max_date)
            if timestamps.max() < max_date:
                issues.append(f"Data ends before max_date: {timestamps.max()} < {max_date}")
        
        # Check for duplicates
        if check_duplicates and timestamps.duplicated().any():
            dup_count = timestamps.duplicated().sum()
            issues.append(f"Found {dup_count} duplicate timestamps")
        
        # Check for gaps
        if check_gaps and len(timestamps) > 1:
            # Determine expected frequency
            freq = pd.infer_freq(timestamps)
            if freq:
                expected = pd.date_range(
                    start=timestamps.min(),
                    end=timestamps.max(),
                    freq=freq
                )
                missing = expected.difference(timestamps)
                if len(missing) > 0:
                    issues.append(f"Found {len(missing)} missing intervals in the time series")
    
    # Check for price anomalies
    price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
    for col in price_cols:
        if df[col].le(0).any():
            issues.append(f"Column '{col}' contains non-positive values")
        
        # Check for large price jumps
        if len(df) > 1:
            returns = df[col].pct_change().dropna()
            large_jumps = returns.abs() > 0.1  # 10% jump
            if large_jumps.any():
                jump_dates = df.loc[large_jumps[large_jumps].index, 'timestamp']
                issues.append(f"Large price jumps in '{col}' on: {', '.join(jump_dates.astype(str).tolist())}")
    
    # Check volume
    if 'volume' in df.columns:
        if df['volume'].lt(0).any():
            issues.append("Negative volume values found")
    
    # Prepare validation result
    result = {
        'is_valid': len(issues) == 0,
        'row_count': len(df),
        'date_range': {
            'min': str(timestamps.min()) if 'timestamp' in df.columns else None,
            'max': str(timestamps.max()) if 'timestamp' in df.columns else None
        },
        'issues': issues
    }
    
    return result

def normalize_data(
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
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure timestamp is the index
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamp')
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Resample to target interval if specified
    if interval:
        # Map interval to pandas frequency string
        freq_map = {
            '1m': '1T',    # 1 minute
            '5m': '5T',    # 5 minutes
            '15m': '15T',  # 15 minutes
            '30m': '30T',  # 30 minutes
            '1h': '1H',    # 1 hour
            '4h': '4H',    # 4 hours
            '1d': '1D',    # 1 day
            '1w': '1W',    # 1 week
        }
        
        if interval not in freq_map:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # Group by symbol and currency before resampling
        group_cols = []
        if 'symbol' in df.columns:
            group_cols.append('symbol')
        if 'currency' in df.columns:
            group_cols.append('currency')
        
        if group_cols:
            dfs = []
            for _, group in df.groupby(group_cols):
                # Resample OHLCV data
                ohlc_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                
                resampled = group.resample(freq_map[interval]).agg(ohlc_dict)
                
                # Forward fill metadata columns
                for col in ['source', 'interval', 'currency', 'symbol']:
                    if col in group.columns:
                        resampled[col] = group[col].iloc[0] if not group.empty else None
                
                dfs.append(resampled)
            
            df = pd.concat(dfs).sort_index()
        else:
            # No grouping needed
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Select only numeric columns for resampling
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            resampled = df[numeric_cols].resample(freq_map[interval]).agg(ohlc_dict)
            
            # Forward fill non-numeric columns
            non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
            if non_numeric_cols:
                non_numeric = df[non_numeric_cols].resample(freq_map[interval]).first()
                df = pd.concat([resampled, non_numeric], axis=1)
            else:
                df = resampled
    
    # Handle missing values
    if fill_method:
        df = df.fillna(method=fill_method).fillna(method='bfill')
    
    # Remove rows with missing values in key columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    required_cols = [col for col in required_cols if col in df.columns]
    df = df.dropna(subset=required_cols)
    
    # Filter by minimum volume
    if 'volume' in df.columns and min_volume > 0:
        df = df[df['volume'] >= min_volume]
    
    # Detect and handle outliers using Z-score
    if outlier_threshold and len(df) > 1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['open', 'high', 'low', 'close']:
                # Calculate returns instead of raw prices for better outlier detection
                returns = df[col].pct_change().dropna()
                if len(returns) > 1:  # Need at least 2 points for std
                    z_scores = (returns - returns.mean()) / returns.std()
                    outliers = np.abs(z_scores) > outlier_threshold
                    if outliers.any():
                        logger.info(f"Found {outliers.sum()} outliers in {col}")
                        # Replace outliers with NaN (they'll be filled by fill_method)
                        df.loc[outliers[outliers].index, col] = np.nan
    
    # Fill any new missing values created by outlier removal
    if fill_method:
        df = df.fillna(method=fill_method).fillna(method='bfill')
    
    # Reset index to make timestamp a column again
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    return df

def convert_currency(
    df: pd.DataFrame,
    target_currency: str,
    base_currency_col: str = 'currency',
    price_cols: List[str] = None,
    volume_col: str = 'volume',
    inplace: bool = False,
    data_manager = None,
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
        data_manager: Optional DataManager instance for fetching FX rates
        
    Returns:
        DataFrame with converted prices and volumes
    """
    if not inplace:
        df = df.copy()
    
    if price_cols is None:
        price_cols = ['open', 'high', 'low', 'close']
    
    # If no data manager is provided, create a temporary one
    if data_manager is None:
        from .data_manager import DataManager
        data_manager = DataManager()
    
    # Get unique currencies in the data
    currencies = df[base_currency_col].unique()
    
    for currency in currencies:
        if currency == target_currency:
            continue  # No conversion needed
            
        # Get FX rates for this currency pair
        fx_rates = data_manager.fetch_fx_rates(
            base_currency=currency,
            target_currency=target_currency,
            start_date=df['timestamp'].min().strftime('%Y-%m-%d'),
            end_date=df['timestamp'].max().strftime('%Y-%m-%d')
        )
        
        if fx_rates.empty:
            logger.warning(f"No FX rates available for {currency}/{target_currency}")
            continue
        
        # Merge FX rates with the original data
        df_merged = pd.merge_asof(
            df[df[base_currency_col] == currency].sort_values('timestamp'),
            fx_rates[['timestamp', 'fx_rate']],
            on='timestamp',
            direction='nearest'
        )
        
        # Convert prices
        for col in price_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col] * df_merged['fx_rate']
        
        # Convert volume (if needed)
        if volume_col and volume_col in df_merged.columns:
            df_merged[volume_col] = df_merged[volume_col] / df_merged['fx_rate']
        
        # Update currency
        df_merged[base_currency_col] = target_currency
        
        # Update the original DataFrame
        df.update(df_merged)
    
    return df

def calculate_returns(
    df: pd.DataFrame,
    price_col: str = 'close',
    return_type: str = 'log',
    periods: int = 1,
    fill_na: bool = True
) -> pd.Series:
    """
    Calculate returns from a price series.
    
    Args:
        df: DataFrame containing the price series
        price_col: Name of the column containing prices
        return_type: Type of returns to calculate ('log' or 'simple')
        periods: Number of periods to shift for the calculation
        fill_na: Whether to fill NA values with 0
        
    Returns:
        Series of returns
    """
    if return_type == 'log':
        returns = np.log(df[price_col] / df[price_col].shift(periods))
    else:  # simple returns
        returns = df[price_col].pct_change(periods=periods)
    
    if fill_na:
        returns = returns.fillna(0)
    
    return returns

def calculate_drawdowns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdowns from an equity curve.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Series of drawdowns as percentages
    """
    # Calculate the running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdowns
    drawdowns = (equity_curve - running_max) / running_max
    
    return drawdowns

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if len(returns) < 2:
        return 0.0
    
    # Annualize the returns and volatility
    annualized_return = excess_returns.mean() * periods_per_year
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    
    # Handle division by zero
    if annualized_vol == 0:
        return 0.0
    
    return annualized_return / annualized_vol

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate the annualized Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 trading days)
        target_return: Target return for downside deviation (default: 0.0)
        
    Returns:
        Annualized Sortino ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate downside returns (returns below target)
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        # If no downside returns, Sortino ratio is infinite
        return float('inf')
    
    # Calculate downside deviation (annualized)
    downside_dev = downside_returns.std() * np.sqrt(periods_per_year)
    
    # Annualized return
    annualized_return = excess_returns.mean() * periods_per_year
    
    # Handle division by zero
    if downside_dev == 0:
        return 0.0
    
    return annualized_return / downside_dev

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the maximum drawdown from an equity curve.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Maximum drawdown as a percentage
    """
    if len(equity_curve) < 2:
        return 0.0
    
    drawdowns = calculate_drawdowns(equity_curve)
    return drawdowns.min() * 100  # Return as percentage

def calculate_calmar_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Calmar ratio (return over max drawdown).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate max drawdown
    max_dd = calculate_max_drawdown(cum_returns)
    
    # Handle division by zero
    if max_dd == 0:
        return 0.0
    
    # Annualized return
    annualized_return = returns.mean() * periods_per_year - risk_free_rate
    
    return annualized_return / (max_dd / 100)  # Convert max_dd from % to decimal
