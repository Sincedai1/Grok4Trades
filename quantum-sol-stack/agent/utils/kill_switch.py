"""
Kill Switch Implementation for QuantumSol Stack

This module provides a kill switch mechanism to immediately halt all trading activities
when triggered by risk management systems or manual intervention.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple
from enum import Enum, auto
import aiohttp
from pathlib import Path
import ccxt

logger = logging.getLogger(__name__)

class KillSwitchStatus(Enum):
    """Status of the kill switch"""
    ACTIVE = auto()
    INACTIVE = auto()
    COOLDOWN = auto()

class KillSwitchReason(Enum):
    """Predefined reasons for kill switch activation"""
    DAILY_LOSS_LIMIT = "Daily loss limit exceeded"
    DRAWDOWN_LIMIT = "Maximum drawdown exceeded"
    MANUAL_OVERRIDE = "Manually triggered by administrator"
    SYSTEM_ERROR = "Critical system error detected"
    MARKET_ABNORMALITY = "Abnormal market conditions detected"
    API_FAILURE = "Exchange API failure"
    NETWORK_ISSUE = "Network connectivity issues detected"
    POSITION_LIMIT = "Position limit exceeded"
    LIQUIDATION_RISK = "High liquidation risk detected"
    SCHEDULED_MAINTENANCE = "Scheduled maintenance"

class KillSwitch:
    """
    Kill Switch implementation for immediate trading halt.
    
    This class provides functionality to:
    1. Immediately cancel all open orders
    2. Close all open positions (if possible)
    3. Prevent new orders from being placed
    4. Notify administrators
    5. Maintain state across restarts
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KillSwitch, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if KillSwitch._initialized:
            return
            
        self.status = KillSwitchStatus.INACTIVE
        self.activation_time: Optional[datetime] = None
        self.reason: Optional[KillSwitchReason] = None
        self.details: Dict = {}
        self._lock = asyncio.Lock()
        self._state_file = Path("kill_switch_state.json")
        self._load_state()
        
        # Initialize exchange connections
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        
        KillSwitch._initialized = True
    
    def _load_state(self):
        """Load kill switch state from disk if available"""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                    self.status = KillSwitchStatus[state.get('status', 'INACTIVE')]
                    if 'activation_time' in state:
                        self.activation_time = datetime.fromisoformat(state['activation_time'])
                    if 'reason' in state:
                        self.reason = KillSwitchReason(state['reason'])
                    self.details = state.get('details', {})
                    
                logger.info(f"Loaded kill switch state: {self.status.name}, "
                          f"Reason: {self.reason.value if self.reason else 'None'}")
        except Exception as e:
            logger.error(f"Error loading kill switch state: {e}")
            # Default to inactive on error
            self.status = KillSwitchStatus.INACTIVE
    
    def _save_state(self):
        """Save current kill switch state to disk"""
        try:
            state = {
                'status': self.status.name,
                'activation_time': self.activation_time.isoformat() if self.activation_time else None,
                'reason': self.reason.value if self.reason else None,
                'details': self.details,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving kill switch state: {e}")
    
    def add_exchange(self, exchange_id: str, exchange: ccxt.Exchange):
        """Add an exchange connection for order management"""
        self.exchanges[exchange_id] = exchange
    
    async def activate(self, 
                      reason: KillSwitchReason, 
                      details: Optional[Dict] = None,
                      force: bool = False) -> bool:
        """
        Activate the kill switch.
        
        Args:
            reason: Reason for activation
            details: Additional details about the activation
            force: If True, will activate even if already active
            
        Returns:
            bool: True if activated, False if already active
        """
        async with self._lock:
            if self.status == KillSwitchStatus.ACTIVE and not force:
                logger.warning(f"Kill switch already active: {self.reason}")
                return False
                
            self.status = KillSwitchStatus.ACTIVE
            self.activation_time = datetime.now(timezone.utc)
            self.reason = reason
            self.details = details or {}
            
            logger.critical(f"KILL SWITCH ACTIVATED - Reason: {reason.value}")
            
            # Save state
            self._save_state()
            
            # Start emergency procedures
            asyncio.create_task(self._emergency_procedures())
            
            return True
    
    async def deactivate(self) -> bool:
        """
        Deactivate the kill switch.
        
        Returns:
            bool: True if deactivated, False if already inactive
        """
        async with self._lock:
            if self.status != KillSwitchStatus.ACTIVE:
                logger.warning("Cannot deactivate - kill switch is not active")
                return False
                
            self.status = KillSwitchStatus.INACTIVE
            self.details['deactivated_at'] = datetime.now(timezone.utc).isoformat()
            
            logger.info("Kill switch deactivated")
            
            # Save state
            self._save_state()
            
            return True
    
    def is_active(self) -> bool:
        """Check if the kill switch is currently active"""
        return self.status == KillSwitchStatus.ACTIVE
    
    async def _emergency_procedures(self):
        """Execute emergency procedures when kill switch is activated"""
        logger.critical("Initiating emergency procedures...")
        
        tasks = []
        
        # Cancel all open orders on all exchanges
        for exchange_id, exchange in self.exchanges.items():
            tasks.append(self._cancel_all_orders(exchange_id, exchange))
        
        # Close all open positions (if supported)
        for exchange_id, exchange in self.exchanges.items():
            tasks.append(self._close_all_positions(exchange_id, exchange))
        
        # Wait for all procedures to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in emergency procedure: {result}")
        
        logger.critical("Emergency procedures completed")
    
    async def _cancel_all_orders(self, exchange_id: str, exchange: ccxt.Exchange) -> Tuple[str, int]:
        """Cancel all open orders on the specified exchange"""
        try:
            logger.info(f"Cancelling all open orders on {exchange_id}...")
            
            # Get all open orders
            orders = await exchange.fetch_open_orders()
            
            if not orders:
                logger.info(f"No open orders to cancel on {exchange_id}")
                return exchange_id, 0
            
            # Cancel all orders
            canceled = 0
            for order in orders:
                try:
                    await exchange.cancel_order(order['id'], order['symbol'])
                    canceled += 1
                    logger.info(f"Cancelled order {order['id']} on {exchange_id}")
                except Exception as e:
                    logger.error(f"Error cancelling order {order['id']}: {e}")
            
            logger.info(f"Cancelled {canceled} orders on {exchange_id}")
            return exchange_id, canceled
            
        except Exception as e:
            logger.error(f"Error in cancel_all_orders for {exchange_id}: {e}")
            return exchange_id, 0
    
    async def _close_all_positions(self, exchange_id: str, exchange: ccxt.Exchange) -> Tuple[str, int]:
        """Close all open positions on the specified exchange"""
        try:
            logger.info(f"Closing all positions on {exchange_id}...")
            
            # Check if exchange supports futures/positions
            if not hasattr(exchange, 'fetch_positions'):
                logger.info(f"Exchange {exchange_id} does not support positions")
                return exchange_id, 0
            
            # Get all open positions
            positions = await exchange.fetch_positions()
            
            # Filter to only positions with non-zero size
            open_positions = [p for p in positions if p.get('contracts', 0) > 0]
            
            if not open_positions:
                logger.info(f"No open positions to close on {exchange_id}")
                return exchange_id, 0
            
            # Close each position with a market order
            closed = 0
            for position in open_positions:
                try:
                    symbol = position['symbol']
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    amount = abs(position['contracts'])
                    
                    # Place market order to close position
                    await exchange.create_market_order(
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        params={'reduceOnly': True}  # Ensure we're only closing
                    )
                    
                    closed += 1
                    logger.info(f"Closed {amount} {symbol} position on {exchange_id}")
                    
                except Exception as e:
                    logger.error(f"Error closing {symbol} position: {e}")
            
            logger.info(f"Closed {closed} positions on {exchange_id}")
            return exchange_id, closed
            
        except Exception as e:
            logger.error(f"Error in close_all_positions for {exchange_id}: {e}")
            return exchange_id, 0
    
    async def get_status(self) -> Dict:
        """Get the current status of the kill switch"""
        return {
            'active': self.is_active(),
            'status': self.status.name,
            'reason': self.reason.value if self.reason else None,
            'activation_time': self.activation_time.isoformat() if self.activation_time else None,
            'details': self.details
        }

# Global instance
kill_switch = KillSwitch()

# Example usage:
# from kill_switch import kill_switch, KillSwitchReason
# 
# # Activate the kill switch
# await kill_switch.activate(
#     reason=KillSwitchReason.DAILY_LOSS_LIMIT,
#     details={'pnl': -3.5, 'threshold': -3.0}
# )
# 
# # Check status
# status = await kill_switch.get_status()
# 
# # Deactivate when ready
# await kill_switch.deactivate()
