import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import asyncio

# Import the components to test
from core.minimal_engine import MinimalTradingBot, TradeSignal
from strategies.simple_ma import SimpleMAStrategy
from risk.basic_limits import BasicRiskManager
from data.simple_logger import SimpleLogger

# Test data
def create_sample_market_data(periods=100, start_price=50000, volatility=0.01):
    """Create sample OHLCV market data for testing"""
    np.random.seed(42)  # For reproducible tests
    
    # Generate random walk for prices
    returns = np.random.normal(0, volatility, periods)
    prices = start_price * (1 + returns).cumprod()
    
    # Create OHLCV data (simplified - using same price for OHLC)
    dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1h')
    data = {
        'open': prices,
        'high': prices * 1.001,  # Slightly higher than close
        'low': prices * 0.999,   # Slightly lower than close
        'close': prices,
        'volume': np.random.uniform(10, 100, periods)
    }
    
    return pd.DataFrame(data, index=dates)

# Fixtures
@pytest.fixture
def sample_market_data():
    """Fixture that provides sample market data"""
    return create_sample_market_data()

@pytest.fixture
def simple_ma_strategy():
    """Fixture that provides a SimpleMAStrategy instance"""
    return SimpleMAStrategy(fast_window=5, slow_window=20, min_confidence=0.6)

@pytest.fixture
def risk_manager():
    """Fixture that provides a BasicRiskManager instance"""
    return BasicRiskManager(
        max_risk_per_trade=0.02,  # 2% risk per trade
        max_daily_loss=0.05,      # 5% max daily loss
        max_position_size=0.1,    # 10% of portfolio
        max_leverage=2.0,         # 2x leverage max
        position_timeout_hours=24  # 24h position timeout
    )

@pytest.fixture
def logger(tmp_path):
    """Fixture that provides a SimpleLogger instance with a temporary directory"""
    log_dir = tmp_path / "logs"
    return SimpleLogger(log_dir=str(log_dir), max_log_days=1)

@pytest.fixture
def mock_exchange():
    """Fixture that provides a mock exchange with basic functionality"""
    class MockExchange:
        def __init__(self):
            self.positions = {}
            self.balance = 10000.0
            
        async def fetch_ticker(self, symbol: str) -> dict:
            return {
                'symbol': symbol,
                'last': 50000.0,
                'bid': 49950.0,
                'ask': 50050.0,
                'timestamp': datetime.utcnow().timestamp()
            }
            
        async def create_order(self, symbol: str, side: str, amount: float, price: float) -> dict:
            cost = amount * price
            self.balance -= cost
            
            if symbol not in self.positions:
                self.positions[symbol] = 0.0
                
            if side.lower() == 'buy':
                self.positions[symbol] += amount
            else:  # sell
                self.positions[symbol] -= amount
                
            return {
                'id': f"mock_order_{int(datetime.utcnow().timestamp() * 1000)}",
                'symbol': symbol,
                'side': side.lower(),
                'amount': amount,
                'price': price,
                'cost': cost,
                'status': 'closed',
                'timestamp': datetime.utcnow().timestamp()
            }
    
    return MockExchange()

# Tests
def test_simple_ma_strategy_signal_generation(simple_ma_strategy, sample_market_data):
    """Test that the SimpleMAStrategy generates signals correctly"""
    # Not enough data for slow MA (needs at least 20 periods)
    with pytest.raises(ValueError):
        simple_ma_strategy.generate_signal(sample_market_data.iloc[:10])
    
    # Generate signal with sufficient data
    signal = simple_ma_strategy.generate_signal(sample_market_data)
    
    # Signal should be either None (no crossover) or a Signal object
    assert signal is None or hasattr(signal, 'action') and signal.action in ['buy', 'sell']
    
    if signal is not None:
        assert signal.price > 0
        assert 0.6 <= signal.confidence <= 1.0
        assert 'crossover' in signal.reason.lower()

def test_risk_manager_position_sizing(risk_manager):
    """Test that position sizing respects risk parameters"""
    account_balance = 10000.0
    entry_price = 50000.0
    stop_loss = 48000.0  # 4% risk if hit
    
    # Calculate position size
    position_size, metrics = risk_manager.calculate_position_size(
        entry_price=entry_price,
        stop_loss=stop_loss,
        account_balance=account_balance
    )
    
    # Check that position size is calculated correctly
    risk_amount = account_balance * 0.02  # 2% risk per trade
    risk_per_share = entry_price - stop_loss
    expected_size = risk_amount / risk_per_share
    
    assert abs(position_size - expected_size) < 1e-8  # Allow for floating point errors
    assert metrics['risk_amount'] == pytest.approx(risk_amount)
    assert metrics['leverage'] <= 2.0  # Should respect max leverage
    
    # Test with confidence < 1.0
    position_size_conf, _ = risk_manager.calculate_position_size(
        entry_price=entry_price,
        stop_loss=stop_loss,
        account_balance=account_balance,
        confidence=0.5  # 50% confidence
    )
    
    # Position size should be smaller with lower confidence
    assert position_size_conf < position_size

def test_risk_manager_daily_limits(risk_manager):
    """Test that daily loss limits are enforced"""
    # Check initial state
    within_limits, message = risk_manager.check_daily_limits()
    assert within_limits is True
    assert "Daily P&L" in message
    
    # Simulate a losing trade that exceeds daily limit
    risk_manager.daily_pnl = -600.0  # 6% loss (over 5% limit)
    
    within_limits, message = risk_manager.check_daily_limits()
    assert within_limits is False
    assert "Daily loss limit reached" in message

@pytest.mark.asyncio
async def test_minimal_trading_bot_lifecycle(mock_exchange, simple_ma_strategy, risk_manager, logger):
    """Test the complete lifecycle of the MinimalTradingBot"""
    # Initialize the bot
    bot = MinimalTradingBot(exchange=mock_exchange, symbol='BTC/USDT')
    
    # Start the bot
    bot.running = True
    
    # Create a trade signal
    signal = TradeSignal(
        symbol='BTC/USDT',
        action='buy',
        price=50000.0,
        size=0.002,  # 0.002 BTC = $100 at $50,000/BTC
        timestamp=datetime.utcnow().timestamp(),
        reason="Test signal"
    )
    
    # Execute a trade
    result = await bot.execute_trade(signal)
    assert result is True
    
    # Verify position was opened
    assert 'BTC/USDT' in bot.positions
    assert bot.positions['BTC/USDT'] > 0
    
    # Stop the bot
    bot.stop()
    assert bot.running is False

def test_logger_functionality(logger, tmp_path):
    """Test that the logger correctly writes and retrieves logs"""
    # Log some trades
    logger.log_trade(
        symbol="BTC/USDT",
        action="buy",
        size=0.01,
        price=50000.0,
        cost=500.0,
        fee=1.25,
        pnl=10.5,
        pnl_pct=2.1,
        balance=10000.0,
        reason="Test trade",
        strategy="SimpleMA",
        order_id="test123"
    )
    
    # Log an event
    logger.log_event(
        event_type="TEST",
        message="This is a test event",
        data={"key": "value"},
        level="INFO"
    )
    
    # Verify logs were written
    assert any(log_file.suffix == '.csv' for log_file in Path(logger.log_dir).glob('trades_*.csv'))
    assert any(log_file.suffix == '.jsonl' for log_file in Path(logger.log_dir).glob('events_*.jsonl'))
    
    # Test log retrieval
    trades = logger.get_recent_trades(limit=1)
    assert len(trades) > 0
    assert trades[0]['symbol'] == 'BTC/USDT'
    
    events = logger.get_recent_events(limit=1, level="INFO")
    assert len(events) > 0
    assert "test event" in events[0]['message'].lower()

# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", "tests/test_basic.py"])
