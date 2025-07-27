"""
Nightly Data Processing and Model Retraining Flow

This module defines the Prefect flow that runs on a schedule to perform
maintenance tasks, data updates, and model retraining.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner

# Local imports
from agent_runner import (
    MonitoringAgent,
    OrchestratorAgent,
    SentimentAgent,
    StrategyAgent,
)
from utils.guardrails_utils import GuardrailEnforcer
from utils.kill_switch import kill_switch
from utils.sentiment_analyzer import sentiment_analyzer
from utils.telegram import MessagePriority, telegram_bot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task(name="backup_database")
async def backup_database() -> bool:
    """Backup the trading database and state.
    
    Returns:
        bool: True if backup was successful, False otherwise
    """
    logger = get_run_logger()
    try:
        # Implementation would depend on your database setup
        # This is a placeholder for the actual backup logic
        logger.info("Starting database backup...")
        await asyncio.sleep(5)  # Simulate backup time
        logger.info("Database backup completed successfully")
        return True
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Database backup failed: %s", e)
        await telegram_bot.send_message(
            f"❌ Database backup failed: {str(e)}",
            priority=MessagePriority.HIGH
        )
        return False

@task(name="update_historical_data")
async def update_historical_data(symbols: List[str] = ["SOL/USDT", "BTC/USDT"]) -> Dict[str, bool]:
    """Update historical price data for the given symbols"""
    logger = get_run_logger()
    results = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Updating historical data for {symbol}...")
            # Implementation would fetch and store historical data
            # This is a placeholder for the actual data update logic
            await asyncio.sleep(2)  # Simulate API calls
            results[symbol] = True
            logger.info(f"Successfully updated data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to update data for {symbol}: {e}")
            results[symbol] = False
    
    return results

@task(name="retrain_models")
async def retrain_models() -> Dict[str, Any]:
    """Retrain machine learning models with the latest data"""
    logger = get_run_logger()
    results = {}
    
    try:
        # Initialize agents
        strategy_agent = StrategyAgent()
        
        # Retrain each model type
        for model_type in ["SMA", "RSI", "LSTM"]:
            try:
                logger.info(f"Retraining {model_type} model...")
                
                # This would call the actual model retraining logic
                # For now, we'll simulate it with a sleep
                await asyncio.sleep(10)  # Simulate training time
                
                # Get model performance metrics (simulated)
                metrics = {
                    "accuracy": np.random.uniform(0.7, 0.95),
                    "sharpe_ratio": np.random.uniform(1.0, 2.5),
                    "win_rate": np.random.uniform(0.5, 0.7),
                    "last_updated": datetime.utcnow().isoformat()
                }
                
                results[model_type] = {"status": "success", "metrics": metrics}
                logger.info(f"Successfully retrained {model_type} model")
                
            except Exception as e:
                logger.error(f"Failed to retrain {model_type} model: {e}")
                results[model_type] = {"status": "failed", "error": str(e)}
    
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")
        return {"status": "error", "message": str(e)}
    
    return results

@task(name="generate_daily_report")
async def generate_daily_report() -> Dict[str, Any]:
    """Generate a daily performance and risk report"""
    logger = get_run_logger()
    
    try:
        # Initialize agents
        monitoring_agent = MonitoringAgent()
        
        # Get performance metrics (simulated)
        report = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "performance": {
                "pnl_24h": np.random.uniform(-2.0, 5.0),  # Simulated PnL
                "win_rate": np.random.uniform(0.5, 0.7),
                "sharpe_ratio": np.random.uniform(1.0, 2.5),
                "total_trades": int(np.random.uniform(10, 50)),
                "open_positions": int(np.random.uniform(0, 5))
            },
            "risk_metrics": {
                "max_drawdown": np.random.uniform(1.0, 5.0),
                "volatility": np.random.uniform(2.0, 8.0),
                "value_at_risk": np.random.uniform(1.0, 3.0)
            },
            "system_health": {
                "status": "operational",
                "last_error": None,
                "uptime": "99.9%"
            },
            "recommendations": [
                "Consider rebalancing portfolio to reduce SOL exposure",
                "High volatility expected tomorrow - adjust position sizing"
            ]
        }
        
        # Save report to file
        report_path = Path("/app/reports") / f"daily_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        logger.info(f"Daily report generated at {report_path}")
        return {"status": "success", "report_path": str(report_path)}
        
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        return {"status": "error", "message": str(e)}

@task(name="check_system_health")
async def check_system_health() -> Dict[str, Any]:
    """Perform system health checks"""
    logger = get_run_logger()
    
    try:
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage("/")
        disk_usage = {
            "total_gb": round(total / (2**30), 1),
            "used_gb": round(used / (2**30), 1),
            "free_gb": round(free / (2**30), 1),
            "percent_used": round(used / total * 100, 1)
        }
        
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        memory_usage = {
            "total_gb": round(memory.total / (1024**3), 1),
            "available_gb": round(memory.available / (1024**3), 1),
            "percent_used": memory.percent
        }
        
        # Check CPU usage
        cpu_usage = {
            "percent_used": psutil.cpu_percent(interval=1),
            "load_avg": [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        }
        
        # Check network connectivity
        import socket
        import aiohttp
        
        network_checks = {
            "internet_connected": False,
            "exchange_api_accessible": False
        }
        
        # Check basic internet connectivity
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            network_checks["internet_connected"] = True
        except (socket.gaierror, socket.timeout):
            logger.warning("No internet connectivity detected")
        
        # Check exchange API connectivity
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.kraken.com/0/public/Time", timeout=10) as resp:
                    network_checks["exchange_api_accessible"] = resp.status == 200
        except Exception as e:
            logger.warning(f"Exchange API check failed: {e}")
        
        # Check kill switch status
        kill_switch_status = await kill_switch.get_status()
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "disk": disk_usage,
            "memory": memory_usage,
            "cpu": cpu_usage,
            "network": network_checks,
            "kill_switch": {
                "active": kill_switch_status["active"],
                "reason": kill_switch_status.get("reason")
            },
            "overall_status": "healthy"
        }
        
        # Determine overall status
        issues = []
        
        if disk_usage["percent_used"] > 90:
            issues.append("High disk usage")
        
        if memory_usage["percent_used"] > 90:
            issues.append("High memory usage")
        
        if cpu_usage["percent_used"] > 90:
            issues.append("High CPU usage")
        
        if not network_checks["internet_connected"]:
            issues.append("No internet connectivity")
        
        if not network_checks["exchange_api_accessible"]:
            issues.append("Exchange API unreachable")
        
        if kill_switch_status["active"]:
            issues.append(f"Kill switch active: {kill_switch_status.get('reason')}")
        
        if issues:
            health_status["overall_status"] = "degraded"
            health_status["issues"] = issues
            
            # Send alert for critical issues
            critical_issues = [
                issue for issue in issues 
                if "disk" in issue.lower() or 
                   "memory" in issue.lower() or
                   "kill switch" in issue.lower()
            ]
            
            if critical_issues:
                await telegram_bot.send_message(
                    f"🚨 <b>System Health Alert</b>\n"
                    f"Critical issues detected:\n"
                    f"• {', '.join(critical_issues)}",
                    priority=MessagePriority.HIGH
                )
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in system health check: {e}")
        return {
            "status": "error",
            "message": str(e),
            "overall_status": "error"
        }

@flow(name="nightly_workflow")
async def nightly_workflow():
    """Main nightly workflow that orchestrates all maintenance tasks"""
    logger = get_run_logger()
    
    try:
        # Initialize Telegram bot
        await telegram_bot.start()
        
        # Notify start of nightly process
        await telegram_bot.send_message(
            "🌙 Starting nightly maintenance process...",
            priority=MessagePriority.LOW
        )
        
        # Run system health check first
        health_status = await check_system_health()
        
        if health_status["overall_status"] != "healthy":
            logger.warning(f"System health issues detected: {health_status.get('issues', [])}")
        
        # Run backup in parallel with other tasks
        backup_task = asyncio.create_task(backup_database())
        
        # Update historical data
        update_task = asyncio.create_task(update_historical_data())
        
        # Wait for both to complete
        backup_result = await backup_task
        update_result = await update_task
        
        # Only proceed with model retraining if data update was successful
        if all(update_result.values()):
            model_results = await retrain_models()
            
            # Generate daily report
            report = await generate_daily_report()
            
            # Send completion notification
            await telegram_bot.send_message(
                "✅ Nightly maintenance completed successfully!\n\n"
                f"• Models retrained: {', '.join(k for k, v in model_results.items() if v.get('status') == 'success')}\n"
                f"• Report generated: {report.get('report_path', 'N/A')}",
                priority=MessagePriority.LOW
            )
        else:
            # Handle data update failure
            failed_symbols = [k for k, v in update_result.items() if not v]
            await telegram_bot.send_message(
                f"❌ Nightly maintenance partially completed. Failed to update data for: {', '.join(failed_symbols)}",
                priority=MessagePriority.HIGH
            )
    
    except Exception as e:
        logger.error(f"Error in nightly workflow: {e}", exc_info=True)
        
        # Send error notification
        await telegram_bot.send_message(
            f"❌ Nightly maintenance failed: {str(e)}",
            priority=MessagePriority.CRITICAL
        )
        
        # Re-raise to mark the flow as failed
        raise
    
    finally:
        # Clean up
        await telegram_bot.stop()

if __name__ == "__main__":
    # Run the flow
    asyncio.run(nightly_workflow())
