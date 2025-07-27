"""
Momentum Strategy for Solana Meme Coins (BONK, etc.)
Optimized for high volatility and rapid price movements
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, RealParameter
from pandas import DataFrame, Series
import talib.abstract as ta
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class MomentumMemeCoin(IStrategy):
    """
    Momentum strategy optimized for Solana meme coins like BONK/USDT.
    
    This strategy looks for strong momentum signals combined with high volume
    and volatility to capture rapid price movements typical of meme coins.
    """
    
    # Strategy interface version - allow new iterations of the strategy class
    INTERFACE_VERSION = 3
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    
    # Can this strategy go short?
    can_short: bool = False
    
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.15,  # 15% profit
        "30": 0.10,  # 10% after 30 minutes
        "60": 0.05,  # 5% after 60 minutes
        "120": 0.01  # 1% after 120 minutes
    }
    
    # Stoploss:
    stoploss = -0.10  # 10% stop loss
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2% trailing stop once in profit
    trailing_stop_positive_offset = 0.04  # Enable trailing after 4% profit
    trailing_only_offset_is_reached = True
    
    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50
    
    # Protections
    stoploss_on_exchange = True
    use_custom_stoploss = True
    
    # Strategy parameters
    # Parameter optimization ranges
    buy_rsi = IntParameter(30, 50, default=40, space='buy')
    sell_rsi = IntParameter(50, 80, default=70, space='sell')
    
    # Volume parameters
    min_volume_usd = DecimalParameter(10000, 500000, default=100000, space='buy')
    volume_spike_multiplier = DecimalParameter(1.5, 5.0, default=2.5, space='buy')
    
    # Momentum parameters
    momentum_period = IntParameter(5, 20, default=10, space='buy')
    momentum_threshold = DecimalParameter(0.5, 3.0, default=1.5, space='buy')
    
    # Volatility parameters
    atr_period = IntParameter(5, 20, default=14, space='buy')
    atr_multiplier = DecimalParameter(1.0, 5.0, default=2.0, space='buy')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        pair = metadata['pair']
        logger.info(f'Populating indicators for {pair}')
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # Volume indicators
        dataframe['volume_ma'] = ta.SMA(dataframe, timeperiod=20, price='volume')
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        
        # Momentum indicators
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=self.momentum_period.value)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        
        # Volume spike detection
        dataframe['volume_spike'] = (dataframe['volume'] > 
                                   (dataframe['volume_ma'] * self.volume_spike_multiplier.value))
        
        # Price momentum
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=self.momentum_period.value)
        
        # Meme coin specific indicators
        dataframe['price_volatility'] = dataframe['close'].pct_change().rolling(window=24).std() * np.sqrt(24)
        
        # Social sentiment (placeholder - would be connected to actual data source)
        dataframe['sentiment_score'] = 0.5  # Placeholder
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        """
        conditions = []
        
        # Primary condition: Strong momentum with volume confirmation
        conditions.append(
            (dataframe['momentum'] > self.momentum_threshold.value) &
            (dataframe['volume'] > (dataframe['volume_ma'] * self.volume_spike_multiplier.value)) &
            (dataframe['rsi'] < self.buy_rsi.value) &
            (dataframe['close'] > dataframe['bb_middle'])  # Price above middle BB
        )
        
        # Secondary condition: Pullback in an uptrend
        conditions.append(
            (dataframe['close'] > dataframe['bb_middle']) &
            (dataframe['close'].shift(1) < dataframe['bb_middle']) &
            (dataframe['volume'] > dataframe['volume_ma']) &
            (dataframe['rsi'] < self.buy_rsi.value)
        )
        
        # Meme coin specific: High volatility and social buzz
        conditions.append(
            (dataframe['price_volatility'] > dataframe['price_volatility'].rolling(72).mean()) &
            (dataframe['sentiment_score'] > 0.7) &
            (dataframe['volume'] > (dataframe['volume_ma'] * 2))
        )
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        """
        conditions = []
        
        # Primary exit: RSI overbought or price hits upper BB
        conditions.append(
            (dataframe['rsi'] > self.sell_rsi.value) |
            (dataframe['close'] > dataframe['bb_upper'])
        )
        
        # Secondary exit: Volume dries up after a run
        conditions.append(
            (dataframe['volume'] < (dataframe['volume_ma'] * 0.5)) &
            (dataframe['close'] > dataframe['bb_middle'])
        )
        
        # Emergency exit: Price drops below lower BB
        conditions.append(
            (dataframe['close'] < dataframe['bb_lower'])
        )
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic for the strategy
        """
        # Let profits run, cut losses quickly
        if current_profit > 0.10:  # 10% profit
            return 0.05  # Tighten stop loss to 5% below current price
        elif current_profit > 0.20:  # 20% profit
            return 0.10  # Lock in at least 10% profit
            
        # Default stoploss
        return self.stoploss
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Customize leverage for each trade. Be careful with high leverage!
        """
        # Use lower leverage for more volatile pairs
        if 'BONK' in pair or 'MEME' in pair:
            return min(3.0, max_leverage)
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
