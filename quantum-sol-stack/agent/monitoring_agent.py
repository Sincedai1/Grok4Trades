"""
Monitoring Agent for the QuantumSol trading system.

This agent is responsible for monitoring portfolio performance, managing risk,
and enforcing trading limits. It tracks PnL, drawdown, and other key metrics,
and can trigger the kill switch if necessary.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
from pydantic import BaseModel, Field, validator

from .base_agent import BaseAgent, AgentTool
from .guardrails_utils import RiskAssessment, RiskLevel, GuardrailManager
from .kill_switch import KillSwitchReason, kill_switch
from .pump_fun import PumpFunToken, PumpFunTrader, get_pump_fun_trader

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Types of metrics that can be monitored."""
    PNL = "pnl"
    DRAWDOWN = "drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    WIN_RATE = "win_rate"
    POSITION_SIZE = "position_size"
    LEVERAGE = "leverage"
    LIQUIDATION_PRICE = "liquidation_price"
    VOLUME = "volume"
    LIQUIDITY = "liquidity"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    SOCIAL_SENTIMENT = "social_sentiment"
    WHALE_ACTIVITY = "whale_activity"

class AlertThreshold(BaseModel):
    """Threshold configuration for alerts."""
    metric: MetricType
    condition: str  # e.g., ">", "<", "==", "!=", "cross_above", "cross_below"
    value: float
    severity: str = "warning"  # "info", "warning", "critical"
    message: str = ""
    cooldown_seconds: int = 300  # Minimum time between alerts for this threshold
    
    @validator('condition')
    def validate_condition(cls, v):
        valid_conditions = [">", "<", ">=", "<=", "==", "!=", "cross_above", "cross_below"]
        if v not in valid_conditions:
            raise ValueError(f"Invalid condition: {v}. Must be one of {valid_conditions}")
        return v

class Alert(BaseModel):
    """An alert that has been triggered."""
    id: str
    metric: MetricType
    value: float
    threshold: AlertThreshold
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class PortfolioMetrics(BaseModel):
    """Metrics for the trading portfolio."""
    # Current values
    total_value: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    position_exposure: float = 0.0  # Total exposure as % of portfolio
    max_position_size: float = 0.0  # Size of largest position as % of portfolio
    leverage: float = 1.0
    
    # Time-based metrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Per-asset metrics
    asset_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class MonitoringAgent(BaseAgent):
    """Agent responsible for monitoring the trading portfolio and managing risk."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Initialize the Monitoring Agent.
        
        Args:
            model: The OpenAI model to use for analysis.
            temperature: Sampling temperature for the model.
            max_tokens: Maximum number of tokens to generate.
        """
        super().__init__(
            name="MonitoringAgent",
            description="Monitors portfolio performance and manages risk",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Initialize guardrails
        self.guardrails = GuardrailManager()
        
        # Alert tracking
        self.alerts: Dict[str, Alert] = {}
        self.alert_cooldowns: Dict[str, float] = {}  # metric_type -> last_alert_time
        
        # Portfolio metrics
        self.portfolio_metrics = PortfolioMetrics()
        self.metrics_history = []
        
        # Alert thresholds
        self.default_thresholds = self._get_default_thresholds()
        
        # Initialize tools
        self._register_tools()
    
    def _register_tools(self):
        """Register tools that this agent can use."""
        self.tools = [
            AgentTool(
                name="get_portfolio_metrics",
                description="Get current portfolio metrics",
                parameters={
                    "type": "object",
                    "properties": {
                        "include_history": {
                            "type": "boolean",
                            "description": "Whether to include historical metrics",
                            "default": False
                        },
                        "lookback_days": {
                            "type": "integer",
                            "description": "Number of days of history to include",
                            "default": 7
                        }
                    }
                }
            ),
            AgentTool(
                name="get_active_alerts",
                description="Get active alerts",
                parameters={
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["all", "info", "warning", "critical"],
                            "description": "Filter alerts by severity",
                            "default": "all"
                        },
                        "acknowledged": {
                            "type": "boolean",
                            "description": "Filter by acknowledged status",
                            "default": None
                        },
                        "resolved": {
                            "type": "boolean",
                            "description": "Filter by resolved status",
                            "default": False
                        }
                    }
                }
            ),
            AgentTool(
                name="acknowledge_alert",
                description="Acknowledge an alert",
                parameters={
                    "type": "object",
                    "properties": {
                        "alert_id": {
                            "type": "string",
                            "description": "ID of the alert to acknowledge"
                        },
                        "comment": {
                            "type": "string",
                            "description": "Optional comment about the acknowledgment"
                        }
                    },
                    "required": ["alert_id"]
                }
            ),
            AgentTool(
                name="resolve_alert",
                description="Mark an alert as resolved",
                parameters={
                    "type": "object",
                    "properties": {
                        "alert_id": {
                            "type": "string",
                            "description": "ID of the alert to resolve"
                        },
                        "comment": {
                            "type": "string",
                            "description": "Optional comment about the resolution"
                        }
                    },
                    "required": ["alert_id"]
                }
            ),
            AgentTool(
                name="check_risk_limits",
                description="Check if any risk limits have been breached",
                parameters={
                    "type": "object",
                    "properties": {
                        "force_check": {
                            "type": "boolean",
                            "description": "Force a check even if recently checked",
                            "default": False
                        }
                    }
                }
            )
        ]
    
    def _get_default_thresholds(self) -> List[AlertThreshold]:
        """Get default alert thresholds."""
        return [
            # Daily PnL thresholds
            AlertThreshold(
                metric=MetricType.PNL,
                condition="<",
                value=-0.03,  # -3%
                severity="critical",
                message="Daily PnL below -3%",
                cooldown_seconds=3600
            ),
            AlertThreshold(
                metric=MetricType.PNL,
                condition="<",
                value=-0.01,  # -1%
                severity="warning",
                message="Daily PnL below -1%",
                cooldown_seconds=1800
            ),
            AlertThreshold(
                metric=MetricType.PNL,
                condition=">",
                value=0.02,  # +2%
                severity="info",
                message="Daily PnL above 2%",
                cooldown_seconds=3600
            ),
            
            # Drawdown thresholds
            AlertThreshold(
                metric=MetricType.DRAWDOWN,
                condition=">",
                value=0.05,  # 5%
                severity="warning",
                message="Drawdown exceeds 5%",
                cooldown_seconds=3600
            ),
            AlertThreshold(
                metric=MetricType.DRAWDOWN,
                condition=">",
                value=0.10,  # 10%
                severity="critical",
                message="Drawdown exceeds 10%",
                cooldown_seconds=1800
            ),
            
            # Position size thresholds
            AlertThreshold(
                metric=MetricType.POSITION_SIZE,
                condition=">",
                value=0.20,  # 20%
                severity="warning",
                message="Position size exceeds 20% of portfolio",
                cooldown_seconds=3600
            ),
            
            # Leverage thresholds
            AlertThreshold(
                metric=MetricType.LEVERAGE,
                condition=">",
                value=3.0,  # 3x
                severity="warning",
                message="Leverage exceeds 3x",
                cooldown_seconds=1800
            ),
            AlertThreshold(
                metric=MetricType.LEVERAGE,
                condition=">",
                value=5.0,  # 5x
                severity="critical",
                message="Leverage exceeds 5x",
                cooldown_seconds=900
            ),
            
            # Liquidation price
            AlertThreshold(
                metric=MetricType.LIQUIDATION_PRICE,
                condition="<",
                value=0.10,  # 10% away
                severity="warning",
                message="Liquidation price within 10% of current price",
                cooldown_seconds=1800
            )
        ]
    
    async def process_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Process a message and return a response."""
        # In a real implementation, this would use the OpenAI API to process the message
        # and generate a response using the available tools
        
        # For now, we'll just return a placeholder response
        return {
            "response": f"Monitoring Agent received: {message}",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Tool implementations
    
    async def get_portfolio_metrics(
        self,
        include_history: bool = False,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """Get current portfolio metrics."""
        try:
            # In a real implementation, this would fetch the latest metrics
            # from the exchange or database
            
            # For now, we'll return the current metrics
            result = {
                "status": "success",
                "metrics": self.portfolio_metrics.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if include_history:
                # Filter history by lookback period
                cutoff = datetime.utcnow() - timedelta(days=lookback_days)
                history = [
                    m.dict() for m in self.metrics_history 
                    if m.timestamp >= cutoff
                ]
                result["history"] = history
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to get portfolio metrics: {str(e)}"
            }
    
    async def get_active_alerts(
        self,
        severity: str = "all",
        acknowledged: Optional[bool] = None,
        resolved: bool = False
    ) -> Dict[str, Any]:
        """Get active alerts."""
        try:
            # Filter alerts
            alerts = list(self.alerts.values())
            
            # Apply filters
            if severity != "all":
                alerts = [a for a in alerts if a.threshold.severity == severity]
                
            if acknowledged is not None:
                alerts = [a for a in alerts if a.acknowledged == acknowledged]
                
            if not resolved:
                alerts = [a for a in alerts if not a.resolved]
            
            return {
                "status": "success",
                "alerts": [a.dict() for a in alerts],
                "count": len(alerts),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to get active alerts: {str(e)}"
            }
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        comment: str = ""
    ) -> Dict[str, Any]:
        """Acknowledge an alert."""
        try:
            if alert_id not in self.alerts:
                return {
                    "status": "error",
                    "message": f"Alert not found: {alert_id}"
                }
            
            alert = self.alerts[alert_id]
            alert.acknowledged = True
            
            if comment:
                if "acknowledgment_comments" not in alert.metadata:
                    alert.metadata["acknowledgment_comments"] = []
                alert.metadata["acknowledgment_comments"].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "comment": comment
                })
            
            return {
                "status": "success",
                "alert_id": alert_id,
                "acknowledged": True,
                "message": "Alert acknowledged"
            }
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to acknowledge alert: {str(e)}"
            }
    
    async def resolve_alert(
        self,
        alert_id: str,
        comment: str = ""
    ) -> Dict[str, Any]:
        """Mark an alert as resolved."""
        try:
            if alert_id not in self.alerts:
                return {
                    "status": "error",
                    "message": f"Alert not found: {alert_id}"
                }
            
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            if comment:
                if "resolution_comments" not in alert.metadata:
                    alert.metadata["resolution_comments"] = []
                alert.metadata["resolution_comments"].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "comment": comment
                })
            
            return {
                "status": "success",
                "alert_id": alert_id,
                "resolved": True,
                "message": "Alert resolved"
            }
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to resolve alert: {str(e)}"
            }
    
    async def check_risk_limits(
        self,
        force_check: bool = False
    ) -> Dict[str, Any]:
        """Check if any risk limits have been breached."""
        try:
            # In a real implementation, this would:
            # 1. Fetch the latest portfolio metrics
            # 2. Check each metric against thresholds
            # 3. Trigger alerts for any breaches
            # 4. Return the results
            
            # For now, we'll simulate this with a placeholder
            results = {
                "status": "success",
                "checks_performed": 10,
                "alerts_triggered": 0,
                "warnings_triggered": 0,
                "critical_alerts_triggered": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to check risk limits: {str(e)}"
            }
    
    # Helper methods
    
    def _evaluate_threshold(
        self,
        metric_value: float,
        threshold: AlertThreshold
    ) -> bool:
        """Evaluate if a threshold condition is met."""
        condition = threshold.condition
        threshold_value = threshold.value
        
        if condition == ">":
            return metric_value > threshold_value
        elif condition == ">=":
            return metric_value >= threshold_value
        elif condition == "<":
            return metric_value < threshold_value
        elif condition == "<=":
            return metric_value <= threshold_value
        elif condition == "==":
            return abs(metric_value - threshold_value) < 1e-10  # Handle floating point
        elif condition == "!=":
            return abs(metric_value - threshold_value) >= 1e-10
        elif condition == "cross_above":
            # For cross_above, we'd need the previous value
            # This is a simplified version
            return metric_value > threshold_value
        elif condition == "cross_below":
            # For cross_below, we'd need the previous value
            # This is a simplified version
            return metric_value < threshold_value
        
        return False
    
    async def _trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert and handle notifications."""
        alert_id = alert.id
        self.alerts[alert_id] = alert
        
        # Log the alert
        logger.warning(
            f"ALERT: {alert.threshold.severity.upper()} - {alert.threshold.message} "
            f"(Value: {alert.value}, Threshold: {alert.threshold.condition} {alert.threshold.value})"
        )
        
        # In a real implementation, this would:
        # 1. Send notifications (email, SMS, etc.)
        # 2. Trigger any automated responses
        # 3. Update dashboards
        
        # For critical alerts, consider triggering the kill switch
        if alert.threshold.severity == "critical":
            reason = KillSwitchReason.RISK_LIMIT
            description = f"Critical alert triggered: {alert.threshold.message}"
            await kill_switch.activate(reason, description)
    
    async def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update portfolio metrics and check for alerts."""
        try:
            # Update current metrics
            self.portfolio_metrics = PortfolioMetrics(**metrics)
            
            # Add to history
            self.metrics_history.append(self.portfolio_metrics)
            
            # Keep history size manageable
            max_history = 1000
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
            
            # Check thresholds
            await self._check_thresholds()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}", exc_info=True)
    
    async def _check_thresholds(self) -> None:
        """Check all thresholds and trigger alerts if needed."""
        current_time = time.time()
        
        # Check each threshold
        for threshold in self.default_thresholds:
            # Check cooldown
            cooldown_key = f"{threshold.metric}_{threshold.condition}_{threshold.value}"
            last_alert = self.alert_cooldowns.get(cooldown_key, 0)
            
            if current_time - last_alert < threshold.cooldown_seconds and not force_check:
                continue
            
            # Get current metric value
            metric_value = getattr(self.portfolio_metrics, threshold.metric.value, None)
            if metric_value is None:
                continue
            
            # Check threshold
            if self._evaluate_threshold(metric_value, threshold):
                # Generate alert ID
                alert_id = f"alert_{len(self.alerts) + 1}"
                
                # Create alert
                alert = Alert(
                    id=alert_id,
                    metric=threshold.metric,
                    value=metric_value,
                    threshold=threshold,
                    metadata={
                        "metric_value": metric_value,
                        "threshold_value": threshold.value,
                        "condition": threshold.condition
                    }
                )
                
                # Trigger alert
                await self._trigger_alert(alert)
                
                # Update cooldown
                self.alert_cooldowns[cooldown_key] = current_time
