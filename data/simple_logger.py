import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import csv
from loguru import logger

class SimpleLogger:
    """
    A simple logger for the trading bot that writes to CSV and JSON files.
    
    This logger is designed to be lightweight and reliable, with minimal dependencies.
    It maintains two types of logs:
    1. Trades log (CSV) - For structured trade data
    2. Events log (JSONL) - For general events and errors
    
    Logs are automatically rotated daily and kept for a configurable number of days.
    """
    
    def __init__(self, log_dir: str = "logs", max_log_days: int = 7):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
            max_log_days: Maximum number of days to keep log files
        """
        self.log_dir = Path(log_dir)
        self.max_log_days = max_log_days
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        current_date = datetime.utcnow().strftime("%Y%m%d")
        self.trades_file = self.log_dir / f"trades_{current_date}.csv"
        self.events_file = self.log_dir / f"events_{current_date}.jsonl"
        
        # Initialize trades CSV with headers if it doesn't exist
        if not self.trades_file.exists():
            self._init_trades_file()
        
        logger.info(f"SimpleLogger initialized. Logging to {self.log_dir.absolute()}")
    
    def _init_trades_file(self):
        """Initialize the trades CSV file with headers"""
        headers = [
            "timestamp", "symbol", "action", "size", "price", "cost", "fee",
            "pnl", "pnl_pct", "balance", "reason", "strategy", "order_id"
        ]
        with open(self.trades_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        size: float,
        price: float,
        cost: float,
        fee: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        balance: Optional[float] = None,
        reason: Optional[str] = None,
        strategy: Optional[str] = None,
        order_id: Optional[str] = None
    ) -> None:
        """
        Log a trade to the trades CSV file.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            action: Trade action ('buy' or 'sell')
            size: Trade size in base currency
            price: Execution price
            cost: Total cost in quote currency (size * price)
            fee: Fee paid for the trade
            pnl: Profit/loss in quote currency
            pnl_pct: Profit/loss as a percentage
            balance: Account balance after the trade
            reason: Reason for the trade
            strategy: Strategy that generated the trade
            order_id: Exchange order ID
        """
        try:
            # Check if we need to rotate the log file
            self._check_rotate_logs()
            
            # Prepare trade data
            trade_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "action": action.lower(),
                "size": size,
                "price": price,
                "cost": cost,
                "fee": fee if fee is not None else "",
                "pnl": pnl if pnl is not None else "",
                "pnl_pct": pnl_pct if pnl_pct is not None else "",
                "balance": balance if balance is not None else "",
                "reason": reason if reason else "",
                "strategy": strategy if strategy else "",
                "order_id": order_id if order_id else ""
            }
            
            # Write to CSV
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trade_data.keys())
                writer.writerow(trade_data)
            
            logger.info(
                f"Trade logged: {action.upper()} {size} {symbol} @ {price:.8f} "
                f"(Cost: {cost:.2f}, PnL: {pnl:.2f} {f'({pnl_pct:.2f}%)' if pnl_pct is not None else ''})"
            )
            
        except Exception as e:
            logger.error(f"Failed to log trade: {str(e)}")
    
    def log_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ) -> None:
        """
        Log an event to the events JSONL file.
        
        Args:
            event_type: Type of event (e.g., 'error', 'warning', 'info')
            message: Event message
            data: Additional event data
            level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        """
        try:
            # Check if we need to rotate the log file
            self._check_rotate_logs()
            
            # Prepare event data
            event_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level.lower(),
                "type": event_type,
                "message": message
            }
            
            # Add additional data if provided
            if data:
                event_data["data"] = data
            
            # Write to JSONL file
            with open(self.events_file, 'a') as f:
                f.write(json.dumps(event_data) + "\n")
            
            # Also log to console using loguru
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(f"[{event_type.upper()}] {message}" + (f" | {data}" if data else ""))
            
        except Exception as e:
            logger.error(f"Failed to log event: {str(e)}")
    
    def _check_rotate_logs(self):
        """Check if log rotation is needed and perform it if necessary"""
        current_date = datetime.utcnow().strftime("%Y%m%d")
        
        # Check if we need to rotate trades log
        if not str(self.trades_file).endswith(f"{current_date}.csv"):
            self.trades_file = self.log_dir / f"trades_{current_date}.csv"
            if not self.trades_file.exists():
                self._init_trades_file()
        
        # Check if we need to rotate events log
        if not str(self.events_file).endswith(f"{current_date}.jsonl"):
            self.events_file = self.log_dir / f"events_{current_date}.jsonl"
        
        # Clean up old log files
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove log files older than max_log_days"""
        if self.max_log_days <= 0:
            return
            
        cutoff_time = time.time() - (self.max_log_days * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.csv"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    logger.debug(f"Removed old log file: {log_file}")
                except Exception as e:
                    logger.error(f"Failed to remove old log file {log_file}: {str(e)}")
        
        for log_file in self.log_dir.glob("*.jsonl"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    logger.debug(f"Removed old log file: {log_file}")
                except Exception as e:
                    logger.error(f"Failed to remove old log file {log_file}: {str(e)}")
    
    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """
        Get recent trades from the log.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries, most recent first
        """
        try:
            trades = []
            
            # Get all trade log files, sorted by date (newest first)
            trade_files = sorted(
                self.log_dir.glob("trades_*.csv"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # Read trades from log files until we have enough
            for trade_file in trade_files:
                with open(trade_file, 'r') as f:
                    reader = csv.DictReader(f)
                    # Skip header if it's the first row
                    if reader.fieldnames and reader.fieldnames[0] == 'timestamp':
                        next(reader, None)
                    # Add trades to the list
                    for row in reader:
                        trades.append(dict(row))
                        if len(trades) >= limit:
                            break
                
                if len(trades) >= limit:
                    break
            
            # Convert numeric fields to appropriate types
            for trade in trades:
                for key in ['size', 'price', 'cost', 'fee', 'pnl', 'pnl_pct', 'balance']:
                    if key in trade and trade[key]:
                        try:
                            trade[key] = float(trade[key])
                        except (ValueError, TypeError):
                            trade[key] = None
            
            return trades[-limit:]  # Return most recent 'limit' trades
            
        except Exception as e:
            logger.error(f"Failed to read trade logs: {str(e)}")
            return []
    
    def get_recent_events(
        self,
        limit: int = 100,
        level: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent events from the log.
        
        Args:
            limit: Maximum number of events to return
            level: Filter by log level (e.g., 'error', 'warning')
            event_type: Filter by event type
            
        Returns:
            List of event dictionaries, most recent first
        """
        try:
            events = []
            
            # Get all event log files, sorted by date (newest first)
            event_files = sorted(
                self.log_dir.glob("events_*.jsonl"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # Read events from log files until we have enough
            for event_file in event_files:
                with open(event_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            
                            # Apply filters
                            if level and event.get('level') != level.lower():
                                continue
                            if event_type and event.get('type') != event_type.lower():
                                continue
                                
                            events.append(event)
                            
                            if len(events) >= limit:
                                break
                                
                        except (json.JSONDecodeError, AttributeError) as e:
                            logger.warning(f"Invalid log entry in {event_file}: {line.strip()}")
                
                if len(events) >= limit:
                    break
            
            return events[-limit:]  # Return most recent 'limit' events
            
        except Exception as e:
            logger.error(f"Failed to read event logs: {str(e)}")
            return []
