from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

@dataclass
class Signal:
    """Data class representing a trading signal"""
    action: str  # 'buy', 'sell', or 'hold'
    price: float
    timestamp: str
    reason: str
    confidence: float  # 0.0 to 1.0
    metadata: Optional[Dict] = None

    def to_dict(self) -> dict:
        """Convert signal to dictionary"""
        return {
            'action': self.action,
            'price': self.price,
            'timestamp': self.timestamp,
            'reason': self.reason,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }

class SimpleMAStrategy:
    """
    Simple Moving Average Crossover Strategy
    
    This strategy generates buy/sell signals based on the crossover of two
    moving averages (fast and slow).
    
    Parameters:
    -----------
    fast_window : int, optional (default=5)
        The window size for the fast moving average
    slow_window : int, optional (default=10)
        The window size for the slow moving average
    min_confidence : float, optional (default=0.6)
        Minimum confidence level (0.0 to 1.0) required to generate a signal
    """
    
    def __init__(self, fast_window: int = 5, slow_window: int = 10, min_confidence: float = 0.6):
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("Window sizes must be positive integers")
        if fast_window >= slow_window:
            raise ValueError("Fast window must be smaller than slow window")
        if not 0 <= min_confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
            
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.min_confidence = min_confidence
        self.previous_signal = None
        
        logger.info(f"Initialized MA Crossover Strategy: {fast_window}/{slow_window}")
    
    def generate_signal(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate trading signal based on moving average crossover
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            DataFrame containing OHLCV data with columns: 'open', 'high', 'low', 'close', 'volume'
            and a DatetimeIndex
            
        Returns:
        --------
        Optional[Signal]
            A Signal object if a valid signal is generated, None otherwise
        """
        try:
            if len(market_data) < self.slow_window + 1:
                logger.warning(f"Not enough data points. Need at least {self.slow_window + 1}, got {len(market_data)}")
                return None
                
            # Calculate moving averages
            close_prices = market_data['close']
            fast_ma = close_prices.rolling(window=self.fast_window).mean()
            slow_ma = close_prices.rolling(window=self.slow_window).mean()
            
            # Get the last two data points for comparison
            current_fast = fast_ma.iloc[-1]
            previous_fast = fast_ma.iloc[-2]
            current_slow = slow_ma.iloc[-1]
            previous_slow = slow_ma.iloc[-2]
            current_price = close_prices.iloc[-1]
            
            # Check for crossover signals
            signal = None
            
            # Bullish signal (fast MA crosses above slow MA)
            if previous_fast <= previous_slow and current_fast > current_slow:
                # Calculate confidence based on the angle of crossover
                fast_slope = (current_fast - previous_fast) / previous_fast if previous_fast > 0 else 0
                slow_slope = (current_slow - previous_slow) / previous_slow if previous_slow > 0 else 0
                confidence = min(0.9, 0.6 + abs(fast_slope - slow_slope) * 10)  # Scale to 0.6-0.9 range
                
                if confidence >= self.min_confidence:
                    signal = Signal(
                        action='buy',
                        price=current_price,
                        timestamp=market_data.index[-1].isoformat(),
                        reason=f'Bullish MA crossover ({self.fast_window}/{self.slow_window})',
                        confidence=confidence,
                        metadata={
                            'fast_ma': current_fast,
                            'slow_ma': current_slow,
                            'fast_slope': fast_slope,
                            'slow_slope': slow_slope
                        }
                    )
            
            # Bearish signal (fast MA crosses below slow MA)
            elif previous_fast >= previous_slow and current_fast < current_slow:
                # Calculate confidence based on the angle of crossover
                fast_slope = (current_fast - previous_fast) / previous_fast if previous_fast > 0 else 0
                slow_slope = (current_slow - previous_slow) / previous_slow if previous_slow > 0 else 0
                confidence = min(0.9, 0.6 + abs(fast_slope - slow_slope) * 10)  # Scale to 0.6-0.9 range
                
                if confidence >= self.min_confidence:
                    signal = Signal(
                        action='sell',
                        price=current_price,
                        timestamp=market_data.index[-1].isoformat(),
                        reason=f'Bearish MA crossover ({self.fast_window}/{self.slow_window})',
                        confidence=confidence,
                        metadata={
                            'fast_ma': current_fast,
                            'slow_ma': current_slow,
                            'fast_slope': fast_slope,
                            'slow_slope': slow_slope
                        }
                    )
            
            # Log the signal if generated
            if signal:
                # Avoid duplicate signals
                if self.previous_signal and self.previous_signal.action == signal.action:
                    logger.debug(f"Skipping duplicate {signal.action} signal")
                    return None
                    
                logger.info(
                    f"Generated {signal.action.upper()} signal at {current_price:.2f} "
                    f"(Fast MA: {current_fast:.2f}, Slow MA: {current_slow:.2f}, "
                    f"Confidence: {signal.confidence:.2f})"
                )
                self.previous_signal = signal
                return signal
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}", exc_info=True)
            return None
    
    def get_required_bars(self) -> int:
        """
        Get the minimum number of bars required for the strategy to work
        
        Returns:
        --------
        int
            Minimum number of bars required
        """
        return max(self.fast_window, self.slow_window) * 2
    
    def get_parameters(self) -> dict:
        """
        Get the current strategy parameters
        
        Returns:
        --------
        dict
            Dictionary containing strategy parameters
        """
        return {
            'strategy': 'SimpleMAStrategy',
            'fast_window': self.fast_window,
            'slow_window': self.slow_window,
            'min_confidence': self.min_confidence
        }
    
    def update_parameters(self, **kwargs):
        """
        Update strategy parameters
        
        Parameters:
        -----------
        **kwargs
            Parameters to update (fast_window, slow_window, min_confidence)
        """
        if 'fast_window' in kwargs:
            self.fast_window = int(kwargs['fast_window'])
        if 'slow_window' in kwargs:
            self.slow_window = int(kwargs['slow_window'])
        if 'min_confidence' in kwargs:
            self.min_confidence = float(kwargs['min_confidence'])
            
        logger.info(f"Updated strategy parameters: {self.get_parameters()}")
        return self.get_parameters()
