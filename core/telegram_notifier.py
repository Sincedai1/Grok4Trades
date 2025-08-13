"""
Telegram Notification Integration for Grok4Trades
Handles all trading alerts and notifications
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import aiohttp
from enum import Enum

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Types of trading alerts"""
    TRADE_SIGNAL = "signal"
    TRADE_EXECUTED = "executed"
    TRADE_CLOSED = "closed"
    RISK_ALERT = "risk"
    SYSTEM_STATUS = "status"
    ERROR = "error"
    DAILY_SUMMARY = "summary"

class TelegramNotifier:
    """Handles Telegram notifications for trading bot"""
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self.session: Optional[aiohttp.ClientSession] = None
        
        if not self.enabled:
            logger.warning("Telegram notifications disabled (missing token or chat_id)")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def send_alert(self, alert_type: AlertType, data: Dict[str, Any]) -> bool:
        """
        Send a formatted alert based on type
        
        Args:
            alert_type: Type of alert to send
            data: Data to populate the template
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
            
        try:
            # Format message based on alert type
            if alert_type == AlertType.TRADE_SIGNAL:
                message = self._format_trade_signal(data)
            elif alert_type == AlertType.TRADE_EXECUTED:
                message = self._format_trade_executed(data)
            elif alert_type == AlertType.TRADE_CLOSED:
                message = self._format_trade_closed(data)
            elif alert_type == AlertType.RISK_ALERT:
                message = self._format_risk_alert(data)
            elif alert_type == AlertType.SYSTEM_STATUS:
                message = self._format_system_status(data)
            elif alert_type == AlertType.ERROR:
                message = self._format_error_alert(data)
            elif alert_type == AlertType.DAILY_SUMMARY:
                message = self._format_daily_summary(data)
            else:
                message = str(data)
            
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send {alert_type.value} alert: {e}")
            return False
    
    async def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("ok", False)
                else:
                    logger.error(f"Telegram API error: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def _format_trade_signal(self, data: Dict[str, Any]) -> str:
        """Format trade signal alert"""
        risk_reward = data.get('risk_reward', 'N/A')
        confidence = data.get('confidence', 0)
        
        return f"""🚀 <b>NEW TRADE SIGNAL</b>

Symbol: <code>{data.get('symbol', 'N/A')}</code>
Side: <b>{data.get('side', 'N/A')}</b>
Entry: ${data.get('entry_price', 0):.2f}
Target: ${data.get('target_price', 0):.2f} ({data.get('profit_pct', 0):.1f}%)
Stop Loss: ${data.get('stop_loss', 0):.2f} ({data.get('loss_pct', 0):.1f}%)

Risk/Reward: {risk_reward}
Confidence: {'⭐' * confidence}/5

<i>Strategy: {data.get('strategy_name', 'Unknown')}</i>"""
    
    def _format_trade_executed(self, data: Dict[str, Any]) -> str:
        """Format trade execution alert"""
        return f"""✅ <b>TRADE EXECUTED</b>

Symbol: <code>{data.get('symbol', 'N/A')}</code>
Side: {data.get('side', 'N/A')}
Amount: {data.get('amount', 0):.4f} (${data.get('notional', 0):.2f})
Price: ${data.get('price', 0):.2f}
Fee: ${data.get('fee', 0):.4f}

Order ID: <code>{data.get('order_id', 'N/A')}</code>
Time: {data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}"""
    
    def _format_trade_closed(self, data: Dict[str, Any]) -> str:
        """Format trade closure alert"""
        pnl = data.get('pnl', 0)
        pnl_pct = data.get('pnl_pct', 0)
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        
        return f"""💰 <b>TRADE CLOSED</b>

Symbol: <code>{data.get('symbol', 'N/A')}</code>
P&L: {pnl_emoji} <b>${pnl:.2f} ({pnl_pct:.1f}%)</b>
Duration: {data.get('duration', 'N/A')}

Entry: ${data.get('entry_price', 0):.2f}
Exit: ${data.get('exit_price', 0):.2f}
Size: {data.get('size', 0):.4f}

Total Trades Today: {data.get('daily_trades', 0)}
Daily P&L: ${data.get('daily_pnl', 0):.2f}"""
    
    def _format_risk_alert(self, data: Dict[str, Any]) -> str:
        """Format risk management alert"""
        return f"""⚠️ <b>RISK ALERT</b>

{data.get('alert_type', 'Risk threshold exceeded')}

Current Drawdown: {data.get('drawdown', 0):.1f}%
Daily Loss: ${data.get('daily_loss', 0):.2f}
Open Positions: {data.get('open_positions', 0)}

Action: {data.get('recommended_action', 'Review positions')}"""
    
    def _format_system_status(self, data: Dict[str, Any]) -> str:
        """Format system status update"""
        status = data.get('status', 'Unknown')
        status_emoji = "🟢" if status == "Running" else "🔴"
        
        return f"""🤖 <b>SYSTEM STATUS</b>

Status: {status_emoji} {status}
Uptime: {data.get('uptime', 'N/A')}
Active Strategies: {data.get('active_strategies', 0)}

Account Balance: ${data.get('balance', 0):.2f}
Free Margin: ${data.get('free_margin', 0):.2f}
Used Margin: ${data.get('used_margin', 0):.2f}

Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    def _format_error_alert(self, data: Dict[str, Any]) -> str:
        """Format error alert"""
        return f"""🚨 <b>ERROR ALERT</b>

Component: {data.get('component', 'Unknown')}
Error: {data.get('error_message', 'Unknown error')}
Time: {data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}

Action Taken: {data.get('action', 'None')}
Impact: {data.get('impact', 'Unknown')}

<i>Check logs for details</i>"""
    
    def _format_daily_summary(self, data: Dict[str, Any]) -> str:
        """Format daily summary"""
        pnl = data.get('total_pnl', 0)
        pnl_pct = data.get('pnl_pct', 0)
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        
        return f"""📊 <b>DAILY SUMMARY</b>
{data.get('date', datetime.now().strftime('%Y-%m-%d'))}

Total Trades: {data.get('total_trades', 0)}
Win Rate: {data.get('win_rate', 0):.1f}%
P&L: {pnl_emoji} ${pnl:.2f} ({pnl_pct:.1f}%)

Best Trade: {data.get('best_trade', 'N/A')} (+${data.get('best_pnl', 0):.2f})
Worst Trade: {data.get('worst_trade', 'N/A')} (-${data.get('worst_pnl', 0):.2f})

Max Drawdown: {data.get('max_drawdown', 0):.1f}%
Sharpe Ratio: {data.get('sharpe_ratio', 0):.2f}

Top Performer: {data.get('top_strategy', 'N/A')}"""

# Convenience functions for quick alerts
async def send_trade_signal(symbol: str, side: str, entry: float, target: float, 
                           stop_loss: float, strategy: str = "Manual", confidence: int = 3):
    """Quick function to send trade signal"""
    async with TelegramNotifier() as notifier:
        profit_pct = ((target - entry) / entry) * 100
        loss_pct = ((entry - stop_loss) / entry) * 100
        risk_reward = f"1:{(profit_pct / loss_pct):.1f}" if loss_pct > 0 else "N/A"
        
        return await notifier.send_alert(AlertType.TRADE_SIGNAL, {
            'symbol': symbol,
            'side': side,
            'entry_price': entry,
            'target_price': target,
            'stop_loss': stop_loss,
            'profit_pct': profit_pct,
            'loss_pct': loss_pct,
            'risk_reward': risk_reward,
            'confidence': confidence,
            'strategy_name': strategy
        })

async def send_error_alert(component: str, error_message: str, 
                          action: str = "Monitoring", impact: str = "Low"):
    """Quick function to send error alert"""
    async with TelegramNotifier() as notifier:
        return await notifier.send_alert(AlertType.ERROR, {
            'component': component,
            'error_message': error_message,
            'action': action,
            'impact': impact,
            'timestamp': datetime.now()
        })

# Example usage
if __name__ == "__main__":
    async def test():
        # Test with a trade signal
        await send_trade_signal(
            symbol="BTC/USDT",
            side="BUY",
            entry=45000,
            target=47000,
            stop_loss=44000,
            strategy="MA Crossover",
            confidence=4
        )
        
        # Test with an error alert
        await send_error_alert(
            component="Risk Manager",
            error_message="Daily loss limit exceeded",
            action="Trading halted",
            impact="High"
        )
    
    asyncio.run(test())
