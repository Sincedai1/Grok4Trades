"""
Meme Coin Pump Detector Strategy
Designed to detect and ride pumps in Solana meme coins like BONK, WIF, etc.
"""
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, RealParameter
from pandas import DataFrame, Series
import talib.abstract as ta
from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import reduce

logger = logging.getLogger(__name__)

class MemeCoinPumpDetector(IStrategy):
    """
    Strategy specifically designed to detect and trade meme coin pumps.
    
    This strategy looks for the unique volume and price action patterns
    that are common in meme coin pumps, particularly on Solana DEXs.
    """
    
    INTERFACE_VERSION = 3
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    
    # Can this strategy go short? (Not recommended for meme coins)
    can_short: bool = False
    
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.30,  # 30% profit
        "15": 0.20,  # 20% after 15 minutes
        "30": 0.10,  # 10% after 30 minutes
        "60": 0.05,  # 5% after 60 minutes
        "120": 0.02  # 2% after 120 minutes
    }
    
    # Stoploss:
    stoploss = -0.20  # 20% stop loss (wider for meme coin volatility)
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05  # 5% trailing stop once in profit
    trailing_stop_positive_offset = 0.10  # Enable trailing after 10% profit
    trailing_only_offset_is_reached = True
    
    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100
    
    # Strategy parameters
    # Volume spike detection
    volume_spike_multiplier = DecimalParameter(2.0, 10.0, default=5.0, space='buy')
    volume_spike_lookback = IntParameter(10, 100, default=50, space='buy')
    
    # Price momentum
    momentum_period = IntParameter(3, 15, default=5, space='buy')
    momentum_threshold = DecimalParameter(1.0, 5.0, default=2.0, space='buy')
    
    # Social metrics (would be connected to actual data source)
    min_social_score = DecimalParameter(0.5, 0.9, default=0.7, space='buy')
    
    # Liquidity parameters
    min_liquidity_usd = DecimalParameter(10000, 500000, default=50000, space='buy')
    
    # Time-based exit
    max_trade_duration_hours = 6  # Close all positions after 6 hours
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators to the given DataFrame
        """
        pair = metadata['pair']
        logger.info(f'Populating indicators for {pair}')
        
        # Basic indicators
        dataframe['sma_10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        
        # Volume analysis
        dataframe['volume_ma'] = ta.SMA(dataframe, timeperiod=20, price='volume')
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # Volume spike detection (z-score)
        volume_mean = dataframe['volume'].rolling(window=self.volume_spike_lookback.value).mean()
        volume_std = dataframe['volume'].rolling(window=self.volume_spike_lookback.value).std()
        dataframe['volume_zscore'] = (dataframe['volume'] - volume_mean) / volume_std
        
        # Price momentum
        dataframe['momentum'] = ta.ROC(dataframe, timeperiod=self.momentum_period.value)
        
        # Volatility indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # Meme coin specific metrics
        dataframe['price_change_1h'] = dataframe['close'].pct_change(periods=12)  # 1h change for 5m candles
        dataframe['price_change_4h'] = dataframe['close'].pct_change(periods=48)  # 4h change
        
        # Liquidity metrics (placeholder - would use actual liquidity data)
        dataframe['liquidity_score'] = 0.7  # Placeholder
        
        # Social metrics (placeholder - would connect to actual data source)
        dataframe['social_volume'] = 0.5  # Placeholder
        dataframe['sentiment_score'] = 0.5  # Placeholder
        
        # Whale activity (large transactions) - placeholder
        dataframe['whale_activity'] = 0.0  # Placeholder
        
        # Pump detection flags
        dataframe['volume_spike'] = dataframe['volume_zscore'] > self.volume_spike_multiplier.value
        dataframe['price_spike'] = dataframe['momentum'] > self.momentum_threshold.value
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on pump detection patterns
        """
        conditions = []
        
        # Primary condition: Volume spike + price momentum
        conditions.append(
            (dataframe['volume_spike']) &
            (dataframe['price_spike']) &
            (dataframe['volume'] > (dataframe['volume_ma'] * self.volume_spike_multiplier.value)) &
            (dataframe['close'] > dataframe['sma_10']) &
            (dataframe['sma_10'] > dataframe['sma_50']) &
            (dataframe['liquidity_score'] * dataframe['close'] * dataframe['volume'] > self.min_liquidity_usd.value)
        )
        
        # Secondary condition: Social volume + price action
        conditions.append(
            (dataframe['social_volume'] > self.min_social_score.value) &
            (dataframe['sentiment_score'] > 0.7) &
            (dataframe['price_change_1h'] > 0.05) &  # 5% price increase in 1h
            (dataframe['volume_ratio'] > 3.0) &
            (dataframe['close'] > dataframe['sma_10'])
        )
        
        # Whale activity condition
        conditions.append(
            (dataframe['whale_activity'] > 0.8) &  # High whale activity
            (dataframe['volume'] > dataframe['volume_ma'] * 3) &
            (dataframe['price_change_1h'] > 0.10)  # 10% price increase in 1h
        )
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on pump exhaustion patterns
        """
        conditions = []
        
        # Exit when volume dries up after a pump
        conditions.append(
            (dataframe['volume'] < (dataframe['volume_ma'] * 0.5)) &
            (dataframe['close'] < dataframe['sma_10'])
        )
        
        # Exit on extreme RSI
        conditions.append(
            (dataframe['rsi'] > 80)  # Overbought
        )
        
        # Exit if price drops below VWAP (if available)
        if 'vmap' in dataframe.columns:
            conditions.append(
                (dataframe['close'] < dataframe['vmap'])
            )
        
        # Time-based exit is handled in custom_exit
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'] = 1
        
        return dataframe
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                   current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Custom exit logic including time-based exits
        """
        # Time-based exit (shorter for meme coins)
        if (current_time - trade.open_date_utc) > timedelta(hours=self.max_trade_duration_hours):
            return 'time_exit'
            
        # Emergency exit for large drawdowns
        if current_profit < -0.15:  # 15% loss
            return 'emergency_exit'
            
        # Let the normal exit conditions work
        return None
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic stop loss that adjusts based on volatility
        """
        # For new trades, use a wider stop to avoid getting stopped out by normal volatility
        if (current_time - trade.open_date_utc) < timedelta(minutes=30):
            return -0.25  # 25% stop loss for first 30 minutes
            
        # After initial period, use a tighter stop based on ATR
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 2:
            return self.stoploss
            
        last_candle = dataframe.iloc[-1].squeeze()
        atr = last_candle['atr']
        
        # Dynamic stop loss based on ATR
        stoploss_price = last_candle['close'] - (atr * 2)  # 2 ATRs below current price
        stoploss_pct = abs(stoploss_price / current_rate - 1)
        
        # Ensure we don't exceed max stoploss
        return min(stoploss_pct, abs(self.stoploss))
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Customize leverage for each trade
        """
        # Be more conservative with leverage on meme coins
        if 'BONK' in pair or 'MEME' in pair or 'WIF' in pair:
            return min(2.0, max_leverage)
        return min(1.5, max_leverage)
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                          **kwargs) -> bool:
        """
        Called right before placing a buy order.
        """
        # Additional checks before entering a trade
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 2:
            return False
            
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Don't enter if we're in a strong downtrend
        if last_candle['close'] < last_candle['sma_50']:
            return False
            
        # Check if we already have an open trade for this pair
        if Trade.get_trades_proxy(is_open=True, pair=pair):
            return False
            
        return True
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        """
        # Log trade exit
        logger.info(
            f"Closing trade for {pair} with {exit_reason}. "
            f"Profit: {trade.calc_profit_ratio(rate):.2%}. "
            f"Duration: {(current_time - trade.open_date_utc)}"
        )
        return True
