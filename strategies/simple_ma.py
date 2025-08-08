from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger

@dataclass
class Signal:
    """Data class representing a trading signal"""
    action: str  # 'buy', 'sell', or 'hold'
    price: float
    timestamp: str
    reason: str
    confidence: float  # 0.0 to 1.0

class SimpleMAStrategy:
    """
    Optimized Moving Average Crossover Strategy using numpy arrays
    """
    
    def __init__(self, fast_window: int = 10, slow_window: int = 20, min_confidence: float = 0.6):
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
        
        # Initialize numpy arrays for price data
        self.prices = np.array([])
        self.fast_ma = np.array([])
        self.slow_ma = np.array([])
        
        logger.info(f"Initialized Optimized MA Crossover Strategy: {fast_window}/{slow_window}")
    
    def _calculate_ma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average using numpy's convolution for better performance"""
        if len(prices) < window:
            return np.array([])
        weights = np.ones(window) / window
        return np.convolve(prices, weights, 'valid')
    
    def _check_crossover(self, fast: np.ndarray, slow: np.ndarray) -> Tuple[bool, bool]:
        """Check for bullish/bearish crossovers between two moving averages"""
        if len(fast) < 2 or len(slow) < 2:
            return False, False
            
        # Get the last two values
        fast_prev, fast_curr = fast[-2], fast[-1]
        slow_prev, slow_curr = slow[-2], slow[-1]
        
        # Check for crossovers
        bullish = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
        bearish = (fast_prev >= slow_prev) and (fast_curr < slow_curr)
        
        return bullish, bearish
    
    def generate_signal(self, market_data: Dict[str, np.ndarray]) -> Optional[Signal]:
        """
        Generate trading signal based on moving average crossover
        
        Args:
            market_data: Dictionary with 'close' key containing numpy array of close prices
            
        Returns:
            Signal object if a valid signal is generated, None otherwise
        """
        try:
            close_prices = market_data['close']
            
            # Update price history (keep only necessary data points)
            self.prices = np.append(self.prices[-self.slow_window*2:], close_prices)
            
            # Calculate moving averages
            self.fast_ma = self._calculate_ma(self.prices, self.fast_window)
            self.slow_ma = self._calculate_ma(self.prices, self.slow_window)
            
            if len(self.fast_ma) < 2 or len(self.slow_ma) < 2:
                logger.warning("Not enough data points for signal generation")
                return None
                
            # Get current values
            current_price = self.prices[-1]
            current_time = str(pd.Timestamp.utcnow())
            
            # Check for crossovers
            bullish, bearish = self._check_crossover(self.fast_ma, self.slow_ma)
            
            # Generate signals
            if bullish:
                confidence = min(0.9, 0.6 + abs(
                    (self.fast_ma[-1] - self.fast_ma[-2]) / (self.fast_ma[-2] + 1e-8) - 
                    (self.slow_ma[-1] - self.slow_ma[-2]) / (self.slow_ma[-2] + 1e-8)
                ) * 10)
                
                if confidence >= self.min_confidence:
                    signal = Signal(
                        action='buy',
                        price=current_price,
                        timestamp=current_time,
                        reason=f'Bullish MA crossover ({self.fast_window}/{self.slow_window})',
                        confidence=confidence
                    )
                    
                    # Avoid duplicate signals
                    if self.previous_signal and self.previous_signal.action == signal.action:
                        logger.debug(f"Skipping duplicate {signal.action} signal")
                        return None
                        
                    logger.info(
                        f"Generated {signal.action.upper()} signal at {current_price:.2f} "
                        f"(Fast MA: {self.fast_ma[-1]:.2f}, Slow MA: {self.slow_ma[-1]:.2f}, "
                        f"Confidence: {signal.confidence:.2f})"
                    )
                    self.previous_signal = signal
                    return signal
                    
            elif bearish:
                confidence = min(0.9, 0.6 + abs(
                    (self.fast_ma[-1] - self.fast_ma[-2]) / (self.fast_ma[-2] + 1e-8) - 
                    (self.slow_ma[-1] - self.slow_ma[-2]) / (self.slow_ma[-2] + 1e-8)
                ) * 10)
                
                if confidence >= self.min_confidence:
                    signal = Signal(
                        action='sell',
                        price=current_price,
                        timestamp=current_time,
                        reason=f'Bearish MA crossover ({self.fast_window}/{self.slow_window})',
                        confidence=confidence
                    )
                    
                    # Avoid duplicate signals
                    if self.previous_signal and self.previous_signal.action == signal.action:
                        logger.debug(f"Skipping duplicate {signal.action} signal")
                        return None
                        
                    logger.info(
                        f"Generated {signal.action.upper()} signal at {current_price:.2f} "
                        f"(Fast MA: {self.fast_ma[-1]:.2f}, Slow MA: {self.slow_ma[-1]:.2f}, "
                        f"Confidence: {signal.confidence:.2f})"
                    )
                    self.previous_signal = signal
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
    
    def get_required_bars(self) -> int:
        """
        Get the minimum number of bars required for the strategy to work
        
        Returns:
        --------
        int
            Minimum number of bars required
        """
        return self.slow_window * 2  # Need enough data for both MAs to stabilize
