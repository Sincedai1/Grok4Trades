import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from loguru import logger
import redis
import inspect
import os
from dataclasses import dataclass, asdict
from typing import Callable

@dataclass
class TradeLog:
    """Data class for trade execution logs"""
    timestamp: str
    symbol: str
    action: str  # 'buy', 'sell', 'cancel', etc.
    price: float
    size: float
    order_id: Optional[str] = None
    status: str = 'created'  # 'created', 'filled', 'canceled', 'rejected'
    reason: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class ErrorLog:
    """Data class for error logs"""
    timestamp: str
    error_type: str
    message: str
    traceback: str
    context: Dict
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""
    timestamp: str
    metric_name: str
    value: float
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

class TradingDebugger:
    """
    Comprehensive debugging and monitoring tool for the trading system.
    
    This class provides structured logging, error tracking, performance monitoring,
    and debugging capabilities for the trading system.
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        redis_key_prefix: str = "debug:",
        max_log_entries: int = 1000,
        log_to_console: bool = True,
        log_to_redis: bool = True
    ):
        """
        Initialize the TradingDebugger
        
        Parameters:
        -----------
        redis_client : redis.Redis, optional
            Redis client for distributed logging. If None, in-memory storage is used.
        redis_key_prefix : str, optional
            Prefix for Redis keys (default: "debug:")
        max_log_entries : int, optional
            Maximum number of log entries to keep in memory (default: 1000)
        log_to_console : bool, optional
            Whether to log to console (default: True)
        log_to_redis : bool, optional
            Whether to log to Redis (default: True)
        """
        self.redis = redis_client
        self.redis_key_prefix = redis_key_prefix
        self.max_log_entries = max_log_entries
        self.log_to_console = log_to_console
        self.log_to_redis = log_to_redis and (redis_client is not None)
        
        # In-memory storage for logs
        self.trade_logs: List[TradeLog] = []
        self.error_logs: List[ErrorLog] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Track execution times
        self._timers: Dict[str, float] = {}
        
        logger.info("Initialized TradingDebugger")
    
    # === Logging Methods ===
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        size: float,
        order_id: Optional[str] = None,
        status: str = 'created',
        reason: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a trade execution
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTC/USDT')
        action : str
            Trade action ('buy', 'sell', 'cancel', etc.)
        price : float
            Execution price
        size : float
            Trade size
        order_id : str, optional
            Exchange order ID
        status : str, optional
            Order status ('created', 'filled', 'canceled', 'rejected')
        reason : str, optional
            Reason for the trade or status change
        metadata : dict, optional
            Additional trade metadata
        """
        log_entry = TradeLog(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            action=action,
            price=price,
            size=size,
            order_id=order_id,
            status=status,
            reason=reason,
            metadata=metadata or {}
        )
        
        # Add to in-memory logs
        self.trade_logs.append(log_entry)
        if len(self.trade_logs) > self.max_log_entries:
            self.trade_logs.pop(0)
        
        # Log to console
        if self.log_to_console:
            log_msg = (
                f"[TRADE] {log_entry.timestamp} | {log_entry.symbol} | {log_entry.action.upper()} | "
                f"{log_entry.size} @ {log_entry.price} | {log_entry.status}"
            )
            if reason:
                log_msg += f" | {reason}"
            logger.info(log_msg)
        
        # Log to Redis
        if self.log_to_redis:
            try:
                trade_key = f"{self.redis_key_prefix}trades:{log_entry.timestamp}"
                self.redis.hmset(trade_key, log_entry.to_dict())
                self.redis.expire(trade_key, timedelta(days=7))  # Keep logs for 7 days
                
                # Add to trade log list
                log_list_key = f"{self.redis_key_prefix}trade_logs"
                self.redis.lpush(log_list_key, trade_key)
                self.redis.ltrim(log_list_key, 0, self.max_log_entries - 1)
            except Exception as e:
                logger.error(f"Failed to log trade to Redis: {str(e)}")
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict] = None,
        trace: Optional[str] = None
    ) -> None:
        """
        Log an error with context
        
        Parameters:
        -----------
        error : Exception
            The exception that was raised
        context : dict, optional
            Additional context about the error
        trace : str, optional
            Custom traceback. If None, uses traceback.format_exc()
        """
        error_trace = trace or traceback.format_exc()
        error_context = context or {}
        
        # Get the calling function name and module
        frame = inspect.currentframe().f_back
        try:
            func_name = frame.f_code.co_name
            module_name = frame.f_globals.get('__name__', 'unknown')
            error_context.update({
                'function': func_name,
                'module': module_name,
                'error_type': error.__class__.__name__,
                'error_message': str(error)
            })
        except Exception:
            pass
        
        log_entry = ErrorLog(
            timestamp=datetime.utcnow().isoformat(),
            error_type=error.__class__.__name__,
            message=str(error),
            traceback=error_trace,
            context=error_context
        )
        
        # Add to in-memory logs
        self.error_logs.append(log_entry)
        if len(self.error_logs) > self.max_log_entries:
            self.error_logs.pop(0)
        
        # Log to console
        if self.log_to_console:
            logger.error(
                f"[ERROR] {log_entry.timestamp} | {log_entry.error_type} | {log_entry.message}\n"
                f"Context: {json.dumps(log_entry.context, indent=2, default=str)}\n"
                f"Traceback:\n{log_entry.traceback}"
            )
        
        # Log to Redis
        if self.log_to_redis:
            try:
                error_key = f"{self.redis_key_prefix}errors:{log_entry.timestamp}"
                self.redis.hmset(error_key, log_entry.to_dict())
                self.redis.expire(error_key, timedelta(days=7))  # Keep logs for 7 days
                
                # Add to error log list
                error_list_key = f"{self.redis_key_prefix}error_logs"
                self.redis.lpush(error_list_key, error_key)
                self.redis.ltrim(error_list_key, 0, self.max_log_entries - 1)
            except Exception as e:
                logger.error(f"Failed to log error to Redis: {str(e)}")
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a performance metric
        
        Parameters:
        -----------
        metric_name : str
            Name of the metric (e.g., 'latency_ms', 'order_execution_time')
        value : float
            Metric value
        metadata : dict, optional
            Additional metadata about the metric
        """
        metric = PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat(),
            metric_name=metric_name,
            value=value,
            metadata=metadata or {}
        )
        
        # Add to in-memory metrics
        self.performance_metrics.append(metric)
        if len(self.performance_metrics) > self.max_log_entries:
            self.performance_metrics.pop(0)
        
        # Log to console
        if self.log_to_console:
            logger.debug(f"[METRIC] {metric_name} = {value} | {json.dumps(metadata or {})}")
        
        # Log to Redis
        if self.log_to_redis:
            try:
                metric_key = f"{self.redis_key_prefix}metrics:{metric_name}:{metric.timestamp}"
                self.redis.hmset(metric_key, metric.to_dict())
                self.redis.expire(metric_key, timedelta(days=1))  # Keep metrics for 1 day
                
                # Add to metric set for this metric name
                metric_set_key = f"{self.redis_key_prefix}metric_names"
                self.redis.sadd(metric_set_key, metric_name)
            except Exception as e:
                logger.error(f"Failed to log metric to Redis: {str(e)}")
    
    # === Performance Timing ===
    
    def start_timer(self, name: str) -> None:
        """Start a named timer"""
        self._timers[name] = time.time()
    
    def stop_timer(self, name: str, log_metric: bool = True) -> float:
        """
        Stop a named timer and return the elapsed time
        
        Parameters:
        -----------
        name : str
            Name of the timer to stop
        log_metric : bool, optional
            Whether to log the elapsed time as a metric (default: True)
            
        Returns:
        --------
        float
            Elapsed time in seconds
        """
        if name not in self._timers:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
            
        elapsed = time.time() - self._timers.pop(name)
        
        if log_metric:
            self.log_metric(
                f"timer_{name}",
                elapsed * 1000,  # Convert to milliseconds
                {'unit': 'ms'}
            )
            
        return elapsed
    
    def time_function(self, func: Callable, *args, **kwargs) -> Any:
        """
        Time a function execution and log the result
        
        Parameters:
        -----------
        func : Callable
            Function to time
        *args, **kwargs
            Arguments to pass to the function
            
        Returns:
        --------
        Any
            The return value of the function
        """
        self.start_timer(func.__name__)
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = self.stop_timer(func.__name__)
            logger.debug(f"Function '{func.__name__}' executed in {elapsed*1000:.2f}ms")
    
    # === Debugging Utilities ===
    
    def get_trade_logs(
        self,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get trade logs with optional filtering
        
        Parameters:
        -----------
        symbol : str, optional
            Filter by symbol
        action : str, optional
            Filter by action ('buy', 'sell', etc.)
        limit : int, optional
            Maximum number of logs to return (default: 100)
            
        Returns:
        --------
        List[Dict]
            List of trade log entries
        """
        logs = self.trade_logs
        
        if symbol:
            logs = [log for log in logs if log.symbol == symbol]
        if action:
            logs = [log for log in logs if log.action == action]
            
        return [log.to_dict() for log in logs[-limit:]]
    
    def get_error_logs(
        self,
        error_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get error logs with optional filtering
        
        Parameters:
        -----------
        error_type : str, optional
            Filter by error type
        limit : int, optional
            Maximum number of logs to return (default: 50)
            
        Returns:
        --------
        List[Dict]
            List of error log entries
        """
        logs = self.error_logs
        
        if error_type:
            logs = [log for log in logs if log.error_type == error_type]
            
        return [log.to_dict() for log in logs[-limit:]]
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get performance metrics with optional filtering
        
        Parameters:
        -----------
        metric_name : str, optional
            Filter by metric name
        start_time : datetime, optional
            Filter metrics after this time
        end_time : datetime, optional
            Filter metrics before this time
        limit : int, optional
            Maximum number of metrics to return (default: 100)
            
        Returns:
        --------
        List[Dict]
            List of metric entries
        """
        metrics = self.performance_metrics
        
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]
        if start_time:
            metrics = [m for m in metrics if datetime.fromisoformat(m.timestamp) >= start_time]
        if end_time:
            metrics = [m for m in metrics if datetime.fromisoformat(m.timestamp) <= end_time]
            
        return [m.to_dict() for m in metrics[-limit:]]
    
    def get_status_summary(self) -> Dict:
        """
        Get a summary of the current system status
        
        Returns:
        --------
        Dict
            Status summary including log counts and recent errors
        """
        recent_errors = [
            {
                'timestamp': log.timestamp,
                'error_type': log.error_type,
                'message': log.message
            }
            for log in self.error_logs[-5:]
        ]
        
        recent_trades = [
            {
                'timestamp': log.timestamp,
                'symbol': log.symbol,
                'action': log.action,
                'price': log.price,
                'size': log.size,
                'status': log.status
            }
            for log in self.trade_logs[-5:]
        ]
        
        return {
            'log_counts': {
                'trades': len(self.trade_logs),
                'errors': len(self.error_logs),
                'metrics': len(self.performance_metrics)
            },
            'recent_errors': recent_errors,
            'recent_trades': recent_trades,
            'active_timers': list(self._timers.keys()),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def clear_logs(self) -> None:
        """Clear all in-memory logs"""
        self.trade_logs = []
        self.error_logs = []
        self.performance_metrics = []
        self._timers = {}
        logger.info("Cleared all debug logs")
