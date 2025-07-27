"""
Volatility Breakout Strategy for Solana Meme Coins
Optimized for capturing large price movements after periods of low volatility
"""
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import reduce

logger = logging.getLogger(__name__)

class VolatilityBreakout(IStrategy):
    """
    Volatility Breakout strategy specifically designed for Solana meme coins.
    
    This strategy identifies periods of low volatility followed by breakouts,
    which are common in meme coins like BONK/USDT.
    """
    
    INTERFACE_VERSION = 3
    
    # Optimal timeframe for the strategy
    timeframe = '15m'
    
    # Can this strategy go short?
    can_short: bool = False
    
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.20,  # 20% profit
        "60": 0.15,  # 15% after 60 minutes
        "120": 0.10,  # 10% after 120 minutes
        "240": 0.05,  # 5% after 240 minutes
        "360": 0.01  # 1% after 360 minutes
    }
    
    # Stoploss:
    stoploss = -0.15  # 15% stop loss
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.03  # 3% trailing stop once in profit
    trailing_stop_positive_offset = 0.06  # Enable trailing after 6% profit
    trailing_only_offset_is_reached = True
    
    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100
    
    # Strategy parameters
    # Parameter optimization ranges
    atr_period = IntParameter(5, 20, default=14, space='buy')
    atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space='buy')
    
    # Volatility contraction parameters
    vol_contraction_period = IntParameter(10, 50, default=20, space='buy')
    vol_contraction_threshold = DecimalParameter(0.1, 0.5, default=0.3, space='buy')
    
    # Volume parameters
    min_volume_usd = DecimalParameter(50000, 500000, default=100000, space='buy')
    volume_spike_multiplier = DecimalParameter(1.5, 5.0, default=2.0, space='buy')
    
    # Breakout confirmation
    breakout_confirmation_candles = IntParameter(1, 5, default=2, space='buy')
    
    # Time-based exit
    max_trade_duration_hours = 24  # Close all positions after 24 hours
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators to the given DataFrame
        """
        pair = metadata['pair']
        logger.info(f'Populating indicators for {pair}')
        
        # Basic indicators
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        
        # Volume indicators
        dataframe['volume_ma'] = ta.SMA(dataframe, timeperiod=20, price='volume')
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # Volatility indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        
        # Calculate volatility contraction (Bollinger Band Width)
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_width'] = (bollinger['upperband'] - bollinger['lowerband']) / bollinger['middleband']
        
        # Historical volatility (standard deviation of returns)
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['hist_volatility'] = dataframe['returns'].rolling(window=20).std() * np.sqrt(365*24*4)  # Annualized
        
        # Donchian Channel for breakout detection
        dataframe['donchian_high'] = dataframe['high'].rolling(window=20).max()
        dataframe['donchian_low'] = dataframe['low'].rolling(window=20).min()
        
        # Keltner Channel for volatility-based channels
        keltner = ta.EMA(dataframe, timeperiod=20)
        atr = ta.ATR(dataframe, timeperiod=20)
        dataframe['keltner_upper'] = keltner + (atr * 2)
        dataframe['keltner_lower'] = keltner - (atr * 2)
        
        # Volume profile
        dataframe['vmap'] = ta.SMA(((dataframe['close'] + dataframe['low'] + dataframe['high']) / 3) * dataframe['volume'], timeperiod=20) / ta.SMA(dataframe['volume'], timeperiod=20)
        
        # Meme coin specific metrics
        dataframe['price_volatility'] = dataframe['close'].pct_change().rolling(window=24).std() * np.sqrt(24)
        dataframe['volume_volatility'] = dataframe['volume'].pct_change().rolling(window=24).std() * np.sqrt(24)
        
        # Social metrics (placeholder - would be connected to actual data source)
        dataframe['social_volume'] = 0.5  # Placeholder
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on volatility breakout patterns
        """
        conditions = []
        
        # Primary condition: Volatility contraction followed by breakout with volume
        conditions.append(
            # Volatility contraction (low BB width)
            (dataframe['bb_width'] < dataframe['bb_width'].rolling(window=self.vol_contraction_period.value).mean() * self.vol_contraction_threshold.value) &
            # Price breaks above Donchian high
            (dataframe['close'] > dataframe['donchian_high'].shift(1)) &
            # Volume confirmation
            (dataframe['volume'] > (dataframe['volume_ma'] * self.volume_spike_multiplier.value)) &
            # Trend filter (optional)
            (dataframe['sma_20'] > dataframe['sma_50']) &
            # Minimum volume threshold
            (dataframe['volume'] * dataframe['close'] > self.min_volume_usd.value)
        )
        
        # Secondary condition: Keltner Channel breakout
        conditions.append(
            (dataframe['close'] > dataframe['keltner_upper']) &
            (dataframe['close'].shift(1) <= dataframe['keltner_upper'].shift(1)) &
            (dataframe['volume'] > dataframe['volume_ma']) &
            (dataframe['volume'] * dataframe['close'] > self.min_volume_usd.value)
        )
        
        # Meme coin specific: High social volume + volatility
        conditions.append(
            (dataframe['social_volume'] > 0.7) &
            (dataframe['price_volatility'] > dataframe['price_volatility'].rolling(72).mean()) &
            (dataframe['volume_ratio'] > 2.0) &
            (dataframe['close'] > dataframe['sma_20'])
        )
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on volatility expansion or time-based rules
        """
        conditions = []
        
        # Exit when price closes below Keltner lower band
        conditions.append(
            (dataframe['close'] < dataframe['keltner_lower'])
        )
        
        # Exit on extreme volatility expansion
        conditions.append(
            (dataframe['bb_width'] > dataframe['bb_width'].rolling(window=50).mean() * 2)
        )
        
        # Exit if volume dries up after entry
        conditions.append(
            (dataframe['volume'] < (dataframe['volume_ma'] * 0.5)) &
            (dataframe['close'] < dataframe['sma_20'])
        )
        
        # Time-based exit (handled in custom_exit)
        
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
        # Time-based exit
        if (current_time - trade.open_date_utc) > timedelta(hours=self.max_trade_duration_hours):
            return 'time_exit'
            
        # Check if we've had a significant move against us
        if current_profit < -0.10:  # 10% loss
            return 'emergency_exit'
            
        # Let the normal exit conditions work
        return None
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic for the strategy
        """
        # Dynamic stop loss based on volatility
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 2:
            return self.stoploss
            
        last_candle = dataframe.iloc[-1].squeeze()
        prev_candle = dataframe.iloc[-2].squeeze()
        
        # Use ATR for dynamic stop loss
        atr = last_candle['atr']
        atr_multiplier = self.atr_multiplier.value
        
        # Calculate stop loss price
        stoploss_price = last_candle['close'] - (atr * atr_multiplier)
        
        # Convert to percentage below current price
        stoploss_pct = abs(stoploss_price / current_rate - 1)
        
        # Ensure we don't exceed max stoploss
        return min(stoploss_pct, abs(self.stoploss))
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Customize leverage for each trade
        """
        # Use lower leverage for more volatile pairs
        if 'BONK' in pair or 'MEME' in pair:
            return min(2.5, max_leverage)
        return min(2.0, max_leverage)
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        """
        # Log trade exit
        logger.info(
            f"Closing trade for {pair} with {exit_reason}. "
            f"Profit: {trade.calc_profit_ratio(rate):.2%}"
        )
        return True
