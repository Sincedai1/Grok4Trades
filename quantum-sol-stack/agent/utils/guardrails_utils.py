"""
Guardrails Utilities for QuantumSol Stack

This module provides utilities for enforcing trading guardrails and safety checks.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, time
import pytz
import logging
from enum import Enum
import ccxt

logger = logging.getLogger(__name__)

class GuardrailViolation(Exception):
    """Exception raised when a guardrail is violated"""
    def __init__(self, guardrail_name: str, message: str):
        self.guardrail_name = guardrail_name
        self.message = message
        super().__init__(f"{guardrail_name} violation: {message}")

class ActionType(str, Enum):
    WARNING = "warning"
    PAUSE_TRADING = "pause_trading"
    EMERGENCY_STOP = "emergency_stop"
    REJECT_ORDER = "reject_order"
    SCALE_DOWN = "scale_down"

class GuardrailEnforcer:
    def __init__(self, config_path: str = "../guardrails_schema.yaml"):
        """Initialize the guardrail enforcer with configuration.
        
        Args:
            config_path: Path to the guardrails YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.violation_count: Dict[str, int] = {}
        self.last_violation: Dict[str, datetime] = {}
        
        # Initialize exchange connections for market data
        self.exchange = ccxt.kraken({
            'apiKey': '',  # Will be set by the main application
            'secret': '',  # Will be set by the main application
            'enableRateLimit': True,
        })
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load guardrails configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load guardrails config: {e}")
            # Return default config if loading fails
            return {
                'risk_parameters': {
                    'max_daily_loss_percent': 3.0,
                    'max_drawdown_percent': 10.0,
                    'max_position_size_usd': 10000,
                    'max_leverage': 3.0,
                    'min_confidence_threshold': 0.7
                }
            }
    
    def check_trading_hours(self) -> Tuple[bool, Optional[str]]:
        """Check if current time is within allowed trading hours."""
        if not self.config.get('trading_hours', {}).get('enabled', False):
            return True, None
        
        try:
            tz = pytz.timezone(self.config['trading_hours'].get('timezone', 'UTC'))
            now = datetime.now(tz)
            current_time = now.time()
            current_weekday = now.weekday()  # Monday is 0, Sunday is 6
            
            for window in self.config['trading_hours'].get('allowed_hours', []):
                start = datetime.strptime(window['start'], '%H:%M').time()
                end = datetime.strptime(window['end'], '%H:%M').time()
                days = window.get('days', list(range(7)))  # Default to all days
                
                if (current_weekday in days and 
                    start <= current_time <= end):
                    return True, None
            
            return False, "Current time is outside allowed trading hours"
            
        except Exception as e:
            logger.error(f"Error checking trading hours: {e}")
            return False, f"Error checking trading hours: {e}"
    
    def check_risk_parameters(self, 
                            pnl_24h: float, 
                            drawdown: float, 
                            position_size_usd: float,
                            leverage: float) -> List[Tuple[str, str, ActionType]]:
        """Check risk parameters against defined guardrails.
        
        Returns:
            List of (guardrail_name, message, action) tuples for any violations
        """
        violations = []
        risk_params = self.config.get('risk_parameters', {})
        
        # Check daily PnL
        max_daily_loss = -abs(risk_params.get('max_daily_loss_percent', 3.0))
        if pnl_24h < max_daily_loss:
            violations.append(
                ("daily_pnl_limit", 
                 f"24h PnL {pnl_24h:.2f}% is below threshold of {max_daily_loss:.2f}%",
                 ActionType.EMERGENCY_STOP)
            )
        
        # Check drawdown
        max_drawdown = -abs(risk_params.get('max_drawdown_percent', 10.0))
        if drawdown < max_drawdown:
            violations.append(
                ("drawdown_limit",
                 f"Drawdown {drawdown:.2f}% is below threshold of {max_drawdown:.2f}%",
                 ActionType.EMERGENCY_STOP)
            )
        
        # Check position size
        max_position_size = risk_params.get('max_position_size_usd', 10000)
        if position_size_usd > max_position_size:
            violations.append(
                ("position_size_limit",
                 f"Position size ${position_size_usd:,.2f} exceeds maximum of ${max_position_size:,.2f}",
                 ActionType.REJECT_ORDER)
            )
        
        # Check leverage
        max_leverage = risk_params.get('max_leverage', 3.0)
        if leverage > max_leverage:
            violations.append(
                ("leverage_limit",
                 f"Leverage {leverage:.1f}x exceeds maximum of {max_leverage:.1f}x",
                 ActionType.REJECT_ORDER)
            )
        
        return violations
    
    async def check_market_conditions(self, symbol: str) -> List[Tuple[str, str, ActionType]]:
        """Check current market conditions for potential issues.
        
        Args:
            symbol: Trading pair symbol (e.g., 'SOL/USDT')
            
        Returns:
            List of (guardrail_name, message, action) tuples for any violations
        """
        violations = []
        
        try:
            # Check 24h volume
            ticker = self.exchange.fetch_ticker(symbol)
            min_volume = self.config.get('asset_restrictions', {}).get('min_volume_24h', 1000000)
            
            if 'quoteVolume' in ticker and ticker['quoteVolume'] < min_volume:
                violations.append(
                    ("volume_restriction",
                     f"24h volume ${ticker['quoteVolume']:,.2f} is below minimum of ${min_volume:,.2f}",
                     ActionType.WARNING)
                )
            
            # Check price volatility (simplified)
            if 'high' in ticker and 'low' in ticker and ticker['last']:
                price_range = ticker['high'] - ticker['low']
                volatility_pct = (price_range / ticker['last']) * 100
                
                for breaker in self.config.get('circuit_breakers', []):
                    if (breaker.get('enabled', False) and 
                        breaker.get('metric') == 'price_volatility_24h' and 
                        volatility_pct > breaker.get('threshold', 15.0)):
                        violations.append(
                            ("volatility_circuit_breaker",
                             f"24h price volatility {volatility_pct:.2f}% exceeds threshold of {breaker['threshold']}%",
                             ActionType.PAUSE_TRADING)
                        )
                        break
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            violations.append(
                ("market_data_error",
                 f"Error checking market conditions: {str(e)}",
                 ActionType.WARNING)
            )
        
        return violations
    
    def check_order_parameters(self, 
                             order_type: str,
                             amount: float,
                             price: float,
                             side: str) -> List[Tuple[str, str, ActionType]]:
        """Validate order parameters against guardrails."""
        violations = []
        order_rules = self.config.get('order_execution', {})
        
        # Check order value
        order_value = amount * price
        max_order_value = order_rules.get('max_order_value_usd', 5000)
        
        if order_value > max_order_value:
            violations.append(
                ("order_value_limit",
                 f"Order value ${order_value:,.2f} exceeds maximum of ${max_order_value:,.2f}",
                 ActionType.REJECT_ORDER)
            )
        
        # Additional order validation logic here
        
        return violations
    
    def should_execute_trade(self, 
                           signal_confidence: float,
                           symbol: str,
                           amount: float,
                           price: float,
                           side: str) -> Tuple[bool, List[str]]:
        """Comprehensive check if a trade should be executed.
        
        Returns:
            Tuple of (should_execute, reasons) where reasons contains any warnings or violations
        """
        reasons = []
        min_confidence = self.config.get('risk_parameters', {}).get('min_confidence_threshold', 0.7)
        
        # Check signal confidence
        if signal_confidence < min_confidence:
            return False, [f"Signal confidence {signal_confidence:.2f} is below minimum threshold {min_confidence:.2f}"]
        
        # Check trading hours
        is_allowed, reason = self.check_trading_hours()
        if not is_allowed and reason:
            reasons.append(reason)
        
        # Check order parameters
        order_violations = self.check_order_parameters(
            order_type="market",  # or get from signal
            amount=amount,
            price=price,
            side=side
        )
        
        # For now, just log order violations as warnings
        for _, msg, _ in order_violations:
            reasons.append(f"Order validation: {msg}")
        
        # If there are any emergency stop conditions, don't execute
        emergency_stops = [v for v in order_violations if v[2] == ActionType.EMERGENCY_STOP]
        if emergency_stops:
            return False, [f"Emergency stop triggered: {emergency_stops[0][1]}"]
        
        # If we get here, the trade is allowed but might have warnings
        return True, reasons
