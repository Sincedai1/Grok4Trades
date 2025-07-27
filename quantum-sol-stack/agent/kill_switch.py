"""
Kill Switch implementation for the QuantumSol trading system.
Allows for immediate halting of all trading activities in case of emergencies.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator
import aiofiles

logger = logging.getLogger(__name__)

class KillSwitchStatus(str, Enum):
    """Status of the kill switch."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    PAUSED = "PAUSED"

class KillSwitchReason(str, Enum):
    """Possible reasons for activating the kill switch."""
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"  # -3% daily loss exceeded
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"  # -10% drawdown exceeded
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"  # Manual activation
    SYSTEM_ERROR = "SYSTEM_ERROR"  # Critical system error
    MARKET_CONDITIONS = "MARKET_CONDITIONS"  # Extreme market conditions
    LIQUIDITY_ISSUE = "LIQUIDITY_ISSUE"  # Liquidity problems detected
    POSITION_LIMIT = "POSITION_LIMIT"  # Position size limit exceeded
    RISK_LIMIT = "RISK_LIMIT"  # Risk limit breached
    OTHER = "OTHER"  # Other reasons

class KillSwitchAction(str, Enum):
    """Actions to take when kill switch is activated."""
    CLOSE_ALL_POSITIONS = "CLOSE_ALL_POSITIONS"  # Close all open positions
    CANCEL_ALL_ORDERS = "CANCEL_ALL_ORDERS"  # Cancel all open orders
    DISABLE_NEW_TRADES = "DISABLE_NEW_TRADES"  # Prevent new trades
    NOTIFY_ADMIN = "NOTIFY_ADMIN"  # Send notification to admin
    LOG_EVENT = "LOG_EVENT"  # Log the event
    
    # Exchange-specific actions
    LIQUIDATE_TO_USD = "LIQUIDATE_TO_USD"  # Convert all to USD
    LIQUIDATE_TO_STABLE = "LIQUIDATE_TO_STABLE"  # Convert to stablecoins
    
    # Risk management actions
    REDUCE_LEVERAGE = "REDUCE_LEVERAGE"  # Reduce leverage
    HEDGE_POSITIONS = "HEDGE_POSITIONS"  # Hedge existing positions

class KillSwitchTrigger(BaseModel):
    """A trigger that can activate the kill switch."""
    reason: KillSwitchReason
    threshold: float  # Value that triggers the kill switch
    current_value: float  # Current value of the metric
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KillSwitchState(BaseModel):
    """Current state of the kill switch."""
    status: KillSwitchStatus = KillSwitchStatus.INACTIVE
    activated_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    reason: Optional[KillSwitchReason] = None
    description: str = ""
    triggers: List[KillSwitchTrigger] = Field(default_factory=list)
    actions_taken: List[KillSwitchAction] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @validator('triggers', pre=True)
    def parse_triggers(cls, v):
        if isinstance(v, str):
            return [KillSwitchTrigger.parse_raw(t) for t in json.loads(v)]
        return v or []
    
    @validator('actions_taken', pre=True)
    def parse_actions(cls, v):
        if isinstance(v, str):
            return [KillSwitchAction(a) for a in json.loads(v)]
        return v or []

class KillSwitch:
    """
    Kill Switch implementation for the QuantumSol trading system.
    
    The kill switch can be triggered by various conditions and will take
    appropriate actions to protect the portfolio.
    """
    
    def __init__(self, state_file: Optional[str] = None):
        """Initialize the kill switch.
        
        Args:
            state_file: Path to a file where the kill switch state will be persisted.
        """
        self.state_file = state_file or "kill_switch_state.json"
        self._state = KillSwitchState()
        self._lock = asyncio.Lock()
        self._callbacks = []
        
        # Load state from file if it exists
        self._load_state()
    
    @property
    def is_active(self) -> bool:
        """Check if the kill switch is currently active."""
        return self._state.status == KillSwitchStatus.ACTIVE
    
    @property
    def status(self) -> KillSwitchStatus:
        """Get the current status of the kill switch."""
        return self._state.status
    
    def add_callback(self, callback):
        """Add a callback to be called when the kill switch is activated or deactivated."""
        self._callbacks.append(callback)
    
    def _load_state(self):
        """Load the kill switch state from disk."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self._state = KillSwitchState(**data)
                    logger.info(f"Loaded kill switch state: {self._state.status}")
        except Exception as e:
            logger.error(f"Error loading kill switch state: {e}")
            self._state = KillSwitchState()
    
    async def _save_state(self):
        """Save the kill switch state to disk."""
        try:
            async with aiofiles.open(self.state_file, 'w') as f:
                await f.write(self._state.json(indent=2))
        except Exception as e:
            logger.error(f"Error saving kill switch state: {e}")
    
    async def _notify_callbacks(self):
        """Notify all registered callbacks of a state change."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._state)
                else:
                    callback(self._state)
            except Exception as e:
                logger.error(f"Error in kill switch callback: {e}")
    
    async def activate(
        self,
        reason: KillSwitchReason,
        description: str = "",
        triggers: Optional[List[KillSwitchTrigger]] = None,
        actions: Optional[List[KillSwitchAction]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Activate the kill switch.
        
        Args:
            reason: The reason for activating the kill switch.
            description: Human-readable description of why the kill switch was activated.
            triggers: List of triggers that caused the activation.
            actions: List of actions to take when activated. If None, default actions will be used.
            metadata: Additional metadata to store with the state.
            
        Returns:
            bool: True if the kill switch was activated, False if it was already active.
        """
        async with self._lock:
            if self._state.status == KillSwitchStatus.ACTIVE:
                logger.warning("Kill switch is already active")
                return False
            
            # Default actions if none provided
            if actions is None:
                actions = [
                    KillSwitchAction.CLOSE_ALL_POSITIONS,
                    KillSwitchAction.CANCEL_ALL_ORDERS,
                    KillSwitchAction.DISABLE_NEW_TRADES,
                    KillSwitchAction.NOTIFY_ADMIN
                ]
            
            # Update state
            self._state.status = KillSwitchStatus.ACTIVE
            self._state.activated_at = datetime.utcnow()
            self._state.deactivated_at = None
            self._state.reason = reason
            self._state.description = description
            self._state.triggers = triggers or []
            self._state.actions_taken = actions
            self._state.metadata = metadata or {}
            
            # Save state
            await self._save_state()
            
            # Log the activation
            logger.critical(
                f"KILL SWITCH ACTIVATED - Reason: {reason.value}\n"
                f"Description: {description}\n"
                f"Triggers: {[t.reason.value for t in triggers] if triggers else 'None'}"
            )
            
            # Notify callbacks
            await self._notify_callbacks()
            
            return True
    
    async def deactivate(self, description: str = "") -> bool:
        """Deactivate the kill switch.
        
        Args:
            description: Reason for deactivating the kill switch.
            
        Returns:
            bool: True if the kill switch was deactivated, False if it was already inactive.
        """
        async with self._lock:
            if self._state.status != KillSwitchStatus.ACTIVE:
                logger.warning("Kill switch is not active")
                return False
            
            # Update state
            self._state.status = KillSwitchStatus.INACTIVE
            self._state.deactivated_at = datetime.utcnow()
            self._state.description = description
            
            # Save state
            await self._save_state()
            
            # Log the deactivation
            logger.info(f"Kill switch deactivated: {description}")
            
            # Notify callbacks
            await self._notify_callbacks()
            
            return True
    
    async def pause(self, description: str = "") -> bool:
        """Pause the kill switch (temporarily disable it).
        
        Args:
            description: Reason for pausing the kill switch.
            
        Returns:
            bool: True if the kill switch was paused, False if it was already paused or inactive.
        """
        async with self._lock:
            if self._state.status != KillSwitchStatus.ACTIVE:
                logger.warning("Cannot pause kill switch: not active")
                return False
            
            self._state.status = KillSwitchStatus.PAUSED
            self._state.description = description
            
            # Save state
            await self._save_state()
            
            logger.warning(f"Kill switch paused: {description}")
            
            return True
    
    async def resume(self) -> bool:
        """Resume the kill switch after pausing.
        
        Returns:
            bool: True if the kill switch was resumed, False if it wasn't paused.
        """
        async with self._lock:
            if self._state.status != KillSwitchStatus.PAUSED:
                logger.warning("Cannot resume kill switch: not paused")
                return False
            
            self._state.status = KillSwitchStatus.ACTIVE
            
            # Save state
            await self._save_state()
            
            logger.info("Kill switch resumed")
            
            return True
    
    def get_state(self) -> KillSwitchState:
        """Get the current state of the kill switch."""
        return self._state.copy(deep=True)
    
    async def check_conditions(
        self,
        metrics: Dict[str, float],
        rules: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[KillSwitchTrigger]]:
        """Check if any kill switch conditions are met based on the given metrics.
        
        Args:
            metrics: Dictionary of metric names to values.
            rules: List of rules defining when to trigger the kill switch.
                   Each rule should have 'metric', 'condition', 'value', and 'reason'.
                   
        Returns:
            Tuple of (should_activate, trigger) where should_activate is a boolean
            indicating if the kill switch should be activated, and trigger is the
            KillSwitchTrigger that caused the activation, or None if no condition was met.
        """
        for rule in rules:
            metric_name = rule['metric']
            condition = rule['condition']
            threshold = rule['value']
            reason = KillSwitchReason(rule['reason'])
            
            if metric_name not in metrics:
                logger.warning(f"Metric '{metric_name}' not found in provided metrics")
                continue
            
            value = metrics[metric_name]
            triggered = False
            
            if condition == '>' and value > threshold:
                triggered = True
            elif condition == '>=' and value >= threshold:
                triggered = True
            elif condition == '<' and value < threshold:
                triggered = True
            elif condition == '<=' and value <= threshold:
                triggered = True
            elif condition == '==' and value == threshold:
                triggered = True
            elif condition == '!=' and value != threshold:
                triggered = True
            
            if triggered:
                trigger = KillSwitchTrigger(
                    reason=reason,
                    threshold=threshold,
                    current_value=value,
                    description=f"{metric_name} {condition} {threshold}",
                    metadata={"rule": rule}
                )
                return True, trigger
        
        return False, None

# Global instance
kill_switch = KillSwitch()

def get_kill_switch() -> KillSwitch:
    """Get the global kill switch instance."""
    return kill_switch
