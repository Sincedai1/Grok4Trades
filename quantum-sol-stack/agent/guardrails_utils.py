"""
Guardrails and safety mechanisms for the QuantumSol agent system.
"""
import logging
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class RiskLevel(str, Enum):
    """Risk levels for trading actions."""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"

class RiskAssessment(BaseModel):
    """Risk assessment for a trading action.
    
    Attributes:
        risk_level: The assessed risk level
        score: Risk score between 0.0 and 1.0
        reasons: List of reasons for the risk assessment
        metadata: Additional metadata about the assessment
    """
    risk_level: RiskLevel
    score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = []
    metadata: Dict[str, Any] = {}
    
    def is_acceptable(self, max_risk: RiskLevel = RiskLevel.MEDIUM) -> bool:
        """Check if the risk is acceptable.
        
        Args:
            max_risk: Maximum acceptable risk level (default: MEDIUM)
            
        Returns:
            bool: True if risk is acceptable, False otherwise
        """
        risk_order = list(RiskLevel)
        return risk_order.index(self.risk_level) <= risk_order.index(max_risk)

class GuardrailResult(BaseModel):
    """Result of applying guardrails to an action.
    
    Attributes:
        allowed: Whether the action is allowed
        risk_assessment: The risk assessment for the action
        modified_action: The modified action if changes were made (optional)
        message: Additional message about the result (optional)
    """
    allowed: bool
    risk_assessment: RiskAssessment
    modified_action: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class Guardrail(BaseModel):
    """Base class for guardrails that validate and potentially modify trading actions.
    
    Attributes:
        name: Name of the guardrail
        description: Description of what the guardrail checks for
    """
    name: str
    description: str
    
    async def check(self, action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check if the action is allowed by this guardrail.
        
        Args:
            action: The trading action to validate
            context: Additional context for the validation (optional)
            
        Returns:
            GuardrailResult: The result of the guardrail check
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass
        """
        raise NotImplementedError

class PositionSizeGuardrail(Guardrail):
    """Guardrail to enforce position size limits as a percentage of portfolio value.
    
    Attributes:
        max_position_size_pct: Maximum allowed position size as a percentage of portfolio value (default: 0.1 or 10%)
    """
    
    def __init__(self, max_position_size_pct: float = 0.1, **kwargs):
        """Initialize the position size guardrail.
        
        Args:
            max_position_size_pct: Maximum position size as a percentage of portfolio value (default: 0.1)
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(
            name="position_size_guardrail",
            description="Limits position size as a percentage of portfolio value",
            **kwargs
        )
        self.max_position_size_pct = max_position_size_pct
    
    async def check(self, action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check if the position size is within acceptable limits.
        
        Args:
            action: The trading action to validate
            context: Additional context including portfolio value (optional)
            
        Returns:
            GuardrailResult: Result of the position size validation
        """
        if action.get("type") != "trade":
            return GuardrailResult(
                allowed=True,
                risk_assessment=RiskAssessment(
                    risk_level=RiskLevel.VERY_LOW,
                    score=0.0,
                    reasons=["Not a trade action"]
                )
            )
        
        portfolio_value = context.get("portfolio_value", 1.0)
        position_size = action.get("amount", 0.0)
        position_size_pct = position_size / portfolio_value if portfolio_value > 0 else 0.0
        
        if position_size_pct > self.max_position_size_pct:
            return GuardrailResult(
                allowed=False,
                risk_assessment=RiskAssessment(
                    risk_level=RiskLevel.HIGH,
                    score=0.8,
                    reasons=[f"Position size {position_size_pct:.1%} exceeds maximum {self.max_position_size_pct:.1%}"]
                ),
                message=f"Position size too large. Maximum allowed: {self.max_position_size_pct:.1%} of portfolio"
            )
        
        return GuardrailResult(
            allowed=True,
            risk_assessment=RiskAssessment(
                risk_level=RiskLevel.LOW,
                score=0.1,
                reasons=["Position size within acceptable limits"],
                metadata={
                    'max_allowed': max_allowed,
                    'requested': position_size,
                    'account_equity': account_equity
                }
            )
        )

class DailyLossLimitGuardrail(Guardrail):
    """Guardrail to enforce daily loss limits.
    
    Attributes:
        max_daily_loss_pct: Maximum allowed daily loss as a percentage of portfolio value (default: 0.03 or 3%)
    """
    
    def __init__(self, max_daily_loss_pct: float = 0.03, **kwargs):
        """Initialize the daily loss limit guardrail.
        
        Args:
            max_daily_loss_pct: Maximum daily loss as a percentage of portfolio value (default: 0.03)
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(
            name="daily_loss_limit_guardrail",
            description="Enforces daily loss limits",
            **kwargs
        )
        self.max_daily_loss_pct = max_daily_loss_pct
    
    async def check(self, action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
        """Check if the daily loss is within acceptable limits.
        
        Args:
            action: The trading action to validate
            context: Additional context including daily PnL percentage (optional)
            
        Returns:
            GuardrailResult: Result of the daily loss limit validation
        """
        if context is None:
            context = {}
            
        daily_pnl_pct = context.get("daily_pnl_pct", 0.0)
        
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            return GuardrailResult(
                allowed=False,
                risk_assessment=RiskAssessment(
                    risk_level=RiskLevel.EXTREME,
                    score=1.0,
                    reasons=[f"Daily PnL {daily_pnl_pct:.2%} exceeds maximum loss of {self.max_daily_loss_pct:.2%}"]
                ),
                message=f"Daily loss limit reached. Current PnL: {daily_pnl_pct:.2%}, Maximum allowed: -{self.max_daily_loss_pct:.2%}"
            )
        
        return GuardrailResult(
            allowed=True,
            risk_assessment=RiskAssessment(
                risk_level=RiskLevel.LOW,
                score=0.0,
                reasons=[f"Daily PnL {daily_pnl_pct:.2%} within limits"]
            )
        )

class PumpFunGuardrail(Guardrail):
    """Guardrail for Pump.fun specific checks."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="pump_fun_guardrail",
            description="Checks for Pump.fun specific risks",
            **kwargs
        )
    
    async def check(self, action: Dict[str, Any], context: Dict[str, Any] = None) -> GuardrailResult:
        """Check if the action is allowed by this guardrail.
        
        Args:
            action: The trading action to validate
            context: Additional context for the validation (optional)
            
        Returns:
            GuardrailResult: The result of the guardrail check
        """
        if context is None:
            context = {}
            
        token_address = action.get("token_address")
        
        # Check if this is a Pump.fun token
        if token_address and self._is_pump_fun_token(token_address):
            # Additional checks for Pump.fun tokens
            liquidity = context.get("liquidity", 0.0)
            volume = context.get("volume_24h", 0.0)
            holders = context.get("holders", 0)
            
            reasons = []
            risk_factors = []
            
            # Check liquidity
            if liquidity < 1000:  # $1000 minimum liquidity
                reasons.append(f"Low liquidity: ${liquidity:,.2f}")
                risk_factors.append(0.8)
            
            # Check volume
            if volume < 5000:  # $5000 minimum 24h volume
                reasons.append(f"Low 24h volume: ${volume:,.2f}")
                risk_factors.append(0.7)
            
            # Check holders
            if holders < 10:  # Minimum 10 holders
                reasons.append(f"Low holder count: {holders}")
                risk_factors.append(0.9)
            
            if risk_factors:
                avg_risk = sum(risk_factors) / len(risk_factors)
                risk_level = self._calculate_risk_level(avg_risk)
                
                return GuardrailResult(
                    allowed=False,
                    risk_assessment=RiskAssessment(
                        risk_level=risk_level,
                        score=avg_risk,
                        reasons=reasons,
                        metadata={
                            "liquidity": liquidity,
                            "volume_24h": volume,
                            "holders": holders
                        }
                    ),
                    message=f"High risk token detected: {', '.join(reasons)}"
                )
        
        return GuardrailResult(
            allowed=True,
            risk_assessment=RiskAssessment(
                risk_level=RiskLevel.LOW,
                score=0.1,
                reasons=["Token passed all checks"]
            )
        )
    
    def _is_pump_fun_token(self, token_address: str) -> bool:
        """Check if a token is a Pump.fun token."""
        # This would typically check against a registry or on-chain data
        # For now, we'll just check if it's a valid Solana address
        return bool(re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", token_address))
    
    def _calculate_risk_level(self, score: float) -> RiskLevel:
        """Convert a risk score to a RiskLevel."""
        if score >= 0.8:
            return RiskLevel.VERY_HIGH
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        elif score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

class GuardrailManager:
    """Manages a collection of guardrails and coordinates their execution.
    
    This class is responsible for managing multiple guardrails, applying them in sequence,
    and aggregating their results. It provides a single entry point for checking if an
    action is allowed by all registered guardrails.
    
    Attributes:
        guardrails: List of guardrail instances to manage
        logger: Logger instance for the manager
    """
    
    def __init__(self, guardrails: Optional[List[Guardrail]] = None):
        """Initialize the guardrail manager with optional initial guardrails.
        
        Args:
            guardrails: Optional list of guardrail instances to manage
        """
        self.guardrails = guardrails or []
        self.logger = logging.getLogger(f"{__name__}.GuardrailManager")
    
    def add_guardrail(self, guardrail: Guardrail) -> None:
        """Add a guardrail to the manager.
        
        Args:
            guardrail: The guardrail instance to add
        """
        if not isinstance(guardrail, Guardrail):
            raise TypeError(f"Expected Guardrail instance, got {type(guardrail).__name__}")
        self.guardrails.append(guardrail)
        self.logger.debug("Added guardrail: %s", guardrail.name)
    
    def add_guardrails(self, guardrails: List[Guardrail]) -> None:
        """Add multiple guardrails to the manager.
        
        Args:
            guardrails: List of guardrail instances to add
        """
        for guardrail in guardrails:
            self.add_guardrail(guardrail)
    
    async def check_action(self, action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check if an action is allowed by all registered guardrails.
        
        This method applies all registered guardrails in sequence. If any guardrail
        rejects the action, processing stops immediately and the rejection is returned.
        
        Args:
            action: The trading action to validate
            context: Additional context for validation (optional)
            
        Returns:
            Dict containing:
                - allowed: bool indicating if the action is allowed
                - reason: str with rejection reason if not allowed
                - details: List of results from all guardrails that were processed
        """
        context = context or {}
        results = []
        
        self.logger.debug("Checking action with %d guardrails", len(self.guardrails))
        
        for guardrail in self.guardrails:
            try:
                result = await guardrail.check(action, context)
                result_dict = {
                    "guardrail": guardrail.name,
                    "allowed": result.allowed,
                    "risk_score": result.risk_assessment.score,
                    "risk_level": result.risk_assessment.risk_level,
                    "reasons": result.risk_assessment.reasons,
                    "message": result.message,
                    "metadata": result.risk_assessment.metadata
                }
                results.append(result_dict)
                
                if not result.allowed:
                    self.logger.warning(
                        "Action blocked by guardrail %s: %s",
                        guardrail.name, 
                        result.message or "No reason provided"
                    )
                    return {
                        "allowed": False,
                        "reason": f"Blocked by {guardrail.name}: {result.message}",
                        "details": results
                    }
                    
            except Exception as e:
                self.logger.error(
                    "Error in guardrail %s: %s", 
                    getattr(guardrail, 'name', 'unknown'), 
                    str(e),
                    exc_info=True
                )
                # Continue with other guardrails even if one fails
                results.append({
                    "guardrail": getattr(guardrail, 'name', 'unknown'),
                    "error": str(e),
                    "allowed": False
                })
        
        self.logger.debug("Action allowed by all guardrails")
        return {
            "allowed": True,
            "details": results
        }

# Default guardrails for the system
DEFAULT_GUARDRAILS = [
    PositionSizeGuardrail(max_position_size_pct=0.1),
    DailyLossLimitGuardrail(max_daily_loss_pct=0.03),
    PumpFunGuardrail()
]
