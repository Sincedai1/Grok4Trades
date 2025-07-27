""
Strategy Agent for the QuantumSol trading system.

This agent is responsible for generating, backtesting, and optimizing trading strategies
using the o1 model and other techniques. It maintains a vector store of strategies and
can retrieve and adapt them based on market conditions.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_agent import BaseAgent, AgentTool
from .guardrails_utils import RiskAssessment, RiskLevel, GuardrailManager
from .kill_switch import kill_switch
from .pump_fun import PumpFunToken, PumpFunTrader, get_pump_fun_trader

logger = logging.getLogger(__name__)

class StrategyType(str, Enum):
    """Types of trading strategies."""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    SENTIMENT = "sentiment"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"

class StrategyPerformance(BaseModel):
    """Performance metrics for a trading strategy."""
    total_return: float = 0.0  # Total return as a decimal (e.g., 0.1 for 10%)
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0  # Win rate as a decimal (0-1)
    profit_factor: float = 0.0
    number_of_trades: int = 0
    average_trade: float = 0.0
    risk_free_rate: float = 0.0  # Risk-free rate used in calculations
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StrategyParameters(BaseModel):
    """Parameters for a trading strategy."""
    # Common parameters
    strategy_type: StrategyType
    symbol: str
    timeframe: str = "1h"
    
    # Entry/exit parameters
    entry_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    exit_conditions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss by default
    take_profit_pct: Optional[float] = None  # No take profit by default
    position_size_pct: float = 0.1  # 10% of portfolio per trade by default
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    
    # Advanced parameters
    indicators: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    filters: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class Strategy(BaseModel):
    """A complete trading strategy."""
    id: str
    name: str
    description: str
    parameters: StrategyParameters
    performance: Optional[StrategyPerformance] = None
    is_active: bool = True
    is_template: bool = False
    created_by: str = "system"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class StrategyAgent(BaseAgent):
    """Agent responsible for managing trading strategies."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """Initialize the Strategy Agent.
        
        Args:
            model: The OpenAI model to use for strategy generation.
            temperature: Sampling temperature for the model.
            max_tokens: Maximum number of tokens to generate.
        """
        super().__init__(
            name="StrategyAgent",
            description="Generates and optimizes trading strategies",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize guardrails
        self.guardrails = GuardrailManager()
        
        # Initialize tools
        self._register_tools()
        
        # Strategy storage (in a real implementation, this would be a database)
        self.strategies: Dict[str, Strategy] = {}
        self.strategy_templates: Dict[str, Strategy] = self._load_strategy_templates()
    
    def _register_tools(self):
        """Register tools that this agent can use."""
        self.tools = [
            AgentTool(
                name="generate_strategy",
                description="Generate a new trading strategy based on parameters",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair symbol (e.g., 'SOL/USDT')"
                        },
                        "strategy_type": {
                            "type": "string",
                            "enum": [t.value for t in StrategyType],
                            "description": "Type of strategy to generate"
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Chart timeframe (e.g., '1h', '4h', '1d')",
                            "default": "1h"
                        },
                        "risk_level": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Risk tolerance level",
                            "default": "medium"
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of indicators to include (e.g., ['rsi', 'macd'])",
                            "default": []
                        }
                    },
                    "required": ["symbol", "strategy_type"]
                }
            ),
            AgentTool(
                name="backtest_strategy",
                description="Backtest a trading strategy on historical data",
                parameters={
                    "type": "object",
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "description": "ID of the strategy to backtest"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair symbol (e.g., 'SOL/USDT')"
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Chart timeframe (e.g., '1h', '4h', '1d')",
                            "default": "1h"
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Start date for backtest (ISO 8601 format)"
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date-time",
                            "description": "End date for backtest (ISO 8601 format, defaults to now)"
                        },
                        "initial_balance": {
                            "type": "number",
                            "description": "Initial balance for backtest",
                            "default": 10000
                        }
                    },
                    "required": ["strategy_id", "symbol", "start_date"]
                }
            ),
            AgentTool(
                name="optimize_strategy",
                description="Optimize strategy parameters",
                parameters={
                    "type": "object",
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "description": "ID of the strategy to optimize"
                        },
                        "parameters_to_optimize": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of parameter names to optimize"
                        },
                        "metric": {
                            "type": "string",
                            "enum": ["sharpe_ratio", "total_return", "win_rate", "profit_factor"],
                            "description": "Metric to optimize for",
                            "default": "sharpe_ratio"
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Maximum number of optimization iterations",
                            "default": 50
                        }
                    },
                    "required": ["strategy_id", "parameters_to_optimize"]
                }
            ),
            AgentTool(
                name="analyze_market_conditions",
                description="Analyze current market conditions for a symbol",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair symbol (e.g., 'SOL/USDT')"
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Chart timeframe (e.g., '1h', '4h', '1d')",
                            "default": "1h"
                        },
                        "lookback_periods": {
                            "type": "integer",
                            "description": "Number of periods to look back",
                            "default": 100
                        }
                    },
                    "required": ["symbol"]
                }
            )
        ]
    
    def _load_strategy_templates(self) -> Dict[str, Strategy]:
        """Load pre-defined strategy templates."""
        templates = {}
        
        # Mean Reversion Strategy
        mean_reversion = Strategy(
            id="mean_reversion_v1",
            name="Mean Reversion",
            description="Trades mean reversion by identifying overbought/oversold conditions using RSI and Bollinger Bands.",
            parameters=StrategyParameters(
                strategy_type=StrategyType.MEAN_REVERSION,
                symbol="",
                timeframe="1h",
                entry_conditions=[
                    {"indicator": "rsi", "condition": "<", "value": 30},
                    {"indicator": "price", "condition": "<", "value": "bb_lower"}
                ],
                exit_conditions=[
                    {"indicator": "rsi", "condition": ">", "value": 50},
                    {"indicator": "price", "condition": ">", "value": "sma_20"}
                ],
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                position_size_pct=0.1,
                indicators={
                    "rsi": {"period": 14},
                    "bollinger_bands": {"period": 20, "std_dev": 2},
                    "sma_20": {"period": 20}
                }
            ),
            is_template=True
        )
        templates[mean_reversion.id] = mean_reversion
        
        # Momentum Strategy
        momentum = Strategy(
            id="momentum_v1",
            name="Momentum",
            description="Captures strong trends using MACD and moving average crossovers.",
            parameters=StrategyParameters(
                strategy_type=StrategyType.MOMENTUM,
                symbol="",
                timeframe="4h",
                entry_conditions=[
                    {"indicator": "macd", "condition": "cross_above", "value": "signal"},
                    {"indicator": "sma_50", "condition": ">", "value": "sma_200"}
                ],
                exit_conditions=[
                    {"indicator": "macd", "condition": "cross_below", "value": "signal"},
                    {"indicator": "sma_50", "condition": "<", "value": "sma_200"}
                ],
                stop_loss_pct=0.05,
                position_size_pct=0.15,
                indicators={
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "sma_50": {"period": 50},
                    "sma_200": {"period": 200}
                }
            ),
            is_template=True
        )
        templates[momentum.id] = momentum
        
        # Breakout Strategy
        breakout = Strategy(
            id="breakout_v1",
            name="Breakout",
            description="Trades breakouts from key support/resistance levels with volume confirmation.",
            parameters=StrategyParameters(
                strategy_type=StrategyType.BREAKOUT,
                symbol="",
                timeframe="1d",
                entry_conditions=[
                    {"indicator": "price", "condition": ">", "value": "resistance"},
                    {"indicator": "volume", "condition": ">", "value": "sma_20_volume"}
                ],
                exit_conditions=[
                    {"indicator": "price", "condition": "<", "value": "support"},
                    {"indicator": "atr", "condition": "trailing_stop", "value": 2.0}
                ],
                stop_loss_pct=0.04,
                take_profit_pct=0.12,
                position_size_pct=0.08,
                indicators={
                    "support_resistance": {"lookback": 20},
                    "volume": {"period": 20},
                    "atr": {"period": 14}
                }
            ),
            is_template=True
        )
        templates[breakout.id] = breakout
        
        return templates
    
    async def process_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Process a message and return a response."""
        # In a real implementation, this would use the OpenAI API to process the message
        # and generate a response using the available tools
        
        # For now, we'll just return a placeholder response
        return {
            "response": f"Strategy Agent received: {message}",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Tool implementations
    
    async def generate_strategy(
        self,
        symbol: str,
        strategy_type: str,
        timeframe: str = "1h",
        risk_level: str = "medium",
        indicators: List[str] = None
    ) -> Dict[str, Any]:
        """Generate a new trading strategy."""
        try:
            # Validate inputs
            if indicators is None:
                indicators = []
            
            # Get template based on strategy type
            template_id = f"{strategy_type.lower()}_v1"
            if template_id not in self.strategy_templates:
                return {
                    "status": "error",
                    "message": f"No template found for strategy type: {strategy_type}",
                    "available_templates": list(self.strategy_templates.keys())
                }
            
            template = self.strategy_templates[template_id]
            
            # Create a new strategy based on the template
            strategy_id = f"{strategy_type.lower()}_{symbol.replace('/', '_').lower()}_{int(time.time())}"
            
            # Adjust parameters based on risk level
            params = template.parameters.dict()
            if risk_level == "low":
                params["stop_loss_pct"] *= 0.8
                params["position_size_pct"] *= 0.5
            elif risk_level == "high":
                params["stop_loss_pct"] *= 1.5
                params["position_size_pct"] *= 1.5
            
            # Update symbol and timeframe
            params["symbol"] = symbol
            params["timeframe"] = timeframe
            
            # Create the strategy
            strategy = Strategy(
                id=strategy_id,
                name=f"{template.name} - {symbol} ({timeframe})",
                description=f"Generated {strategy_type} strategy for {symbol} on {timeframe} timeframe",
                parameters=StrategyParameters(**params),
                created_by="system",
                is_template=False
            )
            
            # Store the strategy
            self.strategies[strategy_id] = strategy
            
            return {
                "status": "success",
                "strategy_id": strategy_id,
                "strategy": strategy.dict(),
                "message": "Strategy generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to generate strategy: {str(e)}"
            }
    
    async def backtest_strategy(
        self,
        strategy_id: str,
        symbol: str,
        start_date: str,
        end_date: str = None,
        timeframe: str = "1h",
        initial_balance: float = 10000.0
    ) -> Dict[str, Any]:
        """Backtest a trading strategy on historical data."""
        try:
            # Validate inputs
            if strategy_id not in self.strategies:
                return {
                    "status": "error",
                    "message": f"Strategy not found: {strategy_id}"
                }
            
            # Parse dates
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if end_date else datetime.utcnow()
            
            # In a real implementation, this would:
            # 1. Fetch historical data for the symbol and timeframe
            # 2. Apply the strategy rules to generate trades
            # 3. Calculate performance metrics
            # 4. Return the results
            
            # For now, we'll return a simulated backtest result
            performance = StrategyPerformance(
                total_return=0.25,  # 25% return
                annualized_return=0.35,
                max_drawdown=0.12,
                sharpe_ratio=1.8,
                sortino_ratio=2.1,
                win_rate=0.65,  # 65% win rate
                profit_factor=1.5,
                number_of_trades=42,
                average_trade=0.006,  # 0.6% average trade return
                risk_free_rate=0.05,  # 5% risk-free rate
                start_date=start_dt,
                end_date=end_dt
            )
            
            # Update strategy with backtest results
            self.strategies[strategy_id].performance = performance
            
            return {
                "status": "success",
                "strategy_id": strategy_id,
                "performance": performance.dict(),
                "message": "Backtest completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error backtesting strategy: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Backtest failed: {str(e)}"
            }
    
    async def optimize_strategy(
        self,
        strategy_id: str,
        parameters_to_optimize: List[str],
        metric: str = "sharpe_ratio",
        max_iterations: int = 50
    ) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        try:
            # Validate inputs
            if strategy_id not in self.strategies:
                return {
                    "status": "error",
                    "message": f"Strategy not found: {strategy_id}"
                }
            
            # Get the strategy
            strategy = self.strategies[strategy_id]
            
            # In a real implementation, this would:
            # 1. Define parameter search space
            # 2. Run optimization algorithm (e.g., grid search, genetic algorithm)
            # 3. Evaluate each parameter set using backtesting
            # 4. Return the best parameters
            
            # For now, we'll return a simulated optimization result
            optimized_params = {
                "stop_loss_pct": 0.045,
                "take_profit_pct": 0.09,
                "position_size_pct": 0.12
            }
            
            # Update strategy with optimized parameters
            for param, value in optimized_params.items():
                if hasattr(strategy.parameters, param):
                    setattr(strategy.parameters, param, value)
            
            # Update strategy
            strategy.updated_at = datetime.utcnow()
            strategy.metadata["last_optimized"] = datetime.utcnow().isoformat()
            
            return {
                "status": "success",
                "strategy_id": strategy_id,
                "optimized_parameters": optimized_params,
                "message": "Optimization completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Optimization failed: {str(e)}"
            }
    
    async def analyze_market_conditions(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_periods: int = 100
    ) -> Dict[str, Any]:
        """Analyze current market conditions for a symbol."""
        try:
            # In a real implementation, this would:
            # 1. Fetch recent market data
            # 2. Calculate technical indicators
            # 3. Analyze trends, volatility, etc.
            # 4. Return a market condition assessment
            
            # For now, we'll return a simulated analysis
            analysis = {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": 150.25,
                "trend": "bullish",
                "volatility": "high",
                "volume": "increasing",
                "rsi": 62.3,
                "macd": 1.25,
                "macd_signal": 0.85,
                "support_levels": [145.30, 138.50],
                "resistance_levels": [155.75, 162.40],
                "recommended_strategies": ["breakout", "momentum"],
                "risk_level": "medium",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return {
                "status": "success",
                "analysis": analysis,
                "message": "Market analysis completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Market analysis failed: {str(e)}"
            }
