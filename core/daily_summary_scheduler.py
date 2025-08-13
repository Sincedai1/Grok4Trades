"""
Automated Daily Summary Scheduler for Grok4Trades
Sends comprehensive daily trading summaries via Telegram
"""

import os
import asyncio
import json
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any
import pandas as pd
import redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from telegram_notifier import TelegramNotifier, AlertType
from custom_alerts import CustomAlerts, ExtendedAlertType

class DailySummaryScheduler:
    """Handles automated daily summary generation and scheduling"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.telegram = TelegramNotifier()
        self.custom_alerts = CustomAlerts(self.telegram)
        self.scheduler = AsyncIOScheduler()
        
        # Redis for storing daily metrics
        self.redis = redis_client or redis.Redis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379'),
            decode_responses=True
        )
        
        # Configure summary times (24-hour format)
        self.summary_times = [
            "08:00",  # Morning summary
            "20:00",  # Evening summary
        ]
        
        # Metrics to track
        self.metrics_keys = {
            'trades': 'daily:trades',
            'pnl': 'daily:pnl',
            'volume': 'daily:volume',
            'errors': 'daily:errors',
            'signals': 'daily:signals'
        }
        
    def start(self):
        """Start the scheduler"""
        # Schedule daily summaries
        for summary_time in self.summary_times:
            hour, minute = map(int, summary_time.split(':'))
            self.scheduler.add_job(
                self._send_daily_summary,
                CronTrigger(hour=hour, minute=minute),
                id=f'daily_summary_{summary_time}',
                name=f'Daily Summary at {summary_time}',
                replace_existing=True
            )
        
        # Schedule hourly status updates
        self.scheduler.add_job(
            self._send_hourly_update,
            CronTrigger(minute=0),  # Every hour on the hour
            id='hourly_update',
            name='Hourly Status Update',
            replace_existing=True
        )
        
        # Schedule market analysis (every 4 hours)
        self.scheduler.add_job(
            self._send_market_analysis,
            CronTrigger(hour='*/4'),
            id='market_analysis',
            name='Market Analysis',
            replace_existing=True
        )
        
        # Schedule performance review (weekly on Sunday)
        self.scheduler.add_job(
            self._send_weekly_performance,
            CronTrigger(day_of_week=6, hour=21, minute=0),  # Sunday 9 PM
            id='weekly_performance',
            name='Weekly Performance Review',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("Daily summary scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("Daily summary scheduler stopped")
    
    async def _send_daily_summary(self):
        """Generate and send comprehensive daily summary"""
        try:
            # Gather daily metrics
            metrics = await self._gather_daily_metrics()
            
            # Generate summary message
            current_hour = datetime.now().hour
            period = "Morning" if current_hour < 12 else "Evening"
            
            message = f"""📊 <b>DAILY {period.upper()} SUMMARY</b>
{datetime.now().strftime('%Y-%m-%d %H:%M')}

<b>Trading Activity:</b>
• Total Trades: {metrics['total_trades']}
• Successful: {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)
• Failed: {metrics['losing_trades']}

<b>Performance:</b>
• Gross Profit: ${metrics['gross_profit']:,.2f}
• Gross Loss: ${metrics['gross_loss']:,.2f}
• Net P&L: <b>${metrics['net_pnl']:,.2f}</b> ({metrics['pnl_pct']:+.2f}%)
• Max Drawdown: {metrics['max_drawdown']:.2f}%

<b>Top Performers:</b>
{self._format_top_trades(metrics['top_trades'])}

<b>Worst Performers:</b>
{self._format_worst_trades(metrics['worst_trades'])}

<b>Market Conditions:</b>
• Volatility: {metrics['market_volatility']}
• Trend: {metrics['market_trend']}
• Volume: {metrics['market_volume']} vs average

<b>Strategy Performance:</b>
{self._format_strategy_performance(metrics['strategies'])}

<b>Risk Metrics:</b>
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}
• Profit Factor: {metrics['profit_factor']:.2f}
• Avg Win: ${metrics['avg_win']:,.2f}
• Avg Loss: ${metrics['avg_loss']:,.2f}

<b>System Health:</b>
• Uptime: {metrics['uptime']}
• Errors: {metrics['error_count']}
• Signals Generated: {metrics['signals_generated']}
• Execution Rate: {metrics['execution_rate']:.1f}%

<b>Recommendations:</b>
{self._generate_recommendations(metrics)}

<i>Next summary at {self._get_next_summary_time()}</i>"""
            
            await self.telegram._send_message(message, parse_mode="HTML")
            
            # Send additional charts/graphs if available
            if metrics.get('has_chart'):
                await self._send_performance_chart(metrics)
            
            # Clear daily metrics after evening summary
            if current_hour >= 20:
                await self._reset_daily_metrics()
                
        except Exception as e:
            logger.error(f"Error sending daily summary: {str(e)}")
            await self._send_error_notification("Daily Summary", str(e))
    
    async def _send_hourly_update(self):
        """Send brief hourly status update"""
        try:
            # Get current metrics
            current_pnl = float(self.redis.get('daily:pnl') or 0)
            trade_count = int(self.redis.get('daily:trades:count') or 0)
            active_positions = await self._get_active_positions()
            
            # Only send if there's activity
            if trade_count > 0 or len(active_positions) > 0:
                emoji = "🟢" if current_pnl >= 0 else "🔴"
                
                message = f"""{emoji} <b>HOURLY UPDATE</b>
{datetime.now().strftime('%H:%M')}

• P&L: ${current_pnl:,.2f}
• Trades: {trade_count}
• Active Positions: {len(active_positions)}
"""
                
                if active_positions:
                    message += "\n<b>Open Positions:</b>\n"
                    for pos in active_positions[:3]:  # Show top 3
                        message += f"• {pos['symbol']}: {pos['side']} ${pos['notional']:,.0f} ({pos['pnl_pct']:+.1f}%)\n"
                
                await self.telegram._send_message(message, parse_mode="HTML")
                
        except Exception as e:
            logger.error(f"Error sending hourly update: {str(e)}")
    
    async def _send_market_analysis(self):
        """Send periodic market analysis"""
        try:
            # Gather market data
            analysis = await self._analyze_market()
            
            await self.custom_alerts.send_market_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Error sending market analysis: {str(e)}")
    
    async def _send_weekly_performance(self):
        """Send weekly performance review"""
        try:
            # Gather weekly metrics
            metrics = await self._gather_weekly_metrics()
            
            message = f"""📈 <b>WEEKLY PERFORMANCE REVIEW</b>
Week of {(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}

<b>Summary:</b>
• Total Trades: {metrics['total_trades']}
• Win Rate: {metrics['win_rate']:.1f}%
• Net P&L: <b>${metrics['net_pnl']:,.2f}</b>
• ROI: {metrics['roi']:.2f}%

<b>Daily Breakdown:</b>
{self._format_daily_breakdown(metrics['daily_pnl'])}

<b>Best Day:</b>
• Date: {metrics['best_day']['date']}
• P&L: +${metrics['best_day']['pnl']:,.2f}
• Trades: {metrics['best_day']['trades']}

<b>Worst Day:</b>
• Date: {metrics['worst_day']['date']}
• P&L: -${metrics['worst_day']['pnl']:,.2f}
• Trades: {metrics['worst_day']['trades']}

<b>Strategy Analysis:</b>
{self._format_weekly_strategy_analysis(metrics['strategies'])}

<b>Risk Analysis:</b>
• Max Drawdown: {metrics['max_drawdown']:.2f}%
• Volatility: {metrics['volatility']:.2f}%
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Sortino Ratio: {metrics['sortino_ratio']:.2f}

<b>Market Comparison:</b>
• BTC Performance: {metrics['btc_performance']:+.2f}%
• Your Performance: {metrics['roi']:+.2f}%
• Alpha: {metrics['alpha']:+.2f}%

<b>Goals for Next Week:</b>
{self._generate_weekly_goals(metrics)}

<i>Keep up the great work! 💪</i>"""
            
            await self.telegram._send_message(message, parse_mode="HTML")
            
            # Send performance chart
            if metrics.get('has_chart'):
                await self._send_weekly_chart(metrics)
                
        except Exception as e:
            logger.error(f"Error sending weekly performance: {str(e)}")
            await self._send_error_notification("Weekly Performance", str(e))
    
    async def _gather_daily_metrics(self) -> Dict[str, Any]:
        """Gather all daily metrics from Redis and trading system"""
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'net_pnl': 0.0,
            'pnl_pct': 0.0,
            'max_drawdown': 0.0,
            'top_trades': [],
            'worst_trades': [],
            'market_volatility': 'Normal',
            'market_trend': 'Neutral',
            'market_volume': 'Average',
            'strategies': {},
            'sharpe_ratio': 0.0,
            'win_loss_ratio': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'uptime': 'N/A',
            'error_count': 0,
            'signals_generated': 0,
            'execution_rate': 0.0,
            'has_chart': False
        }
        
        try:
            # Get trades from Redis
            trades_data = self.redis.get('daily:trades:data')
            if trades_data:
                trades = json.loads(trades_data)
                metrics['total_trades'] = len(trades)
                
                # Calculate trade statistics
                winning = [t for t in trades if t.get('pnl', 0) > 0]
                losing = [t for t in trades if t.get('pnl', 0) < 0]
                
                metrics['winning_trades'] = len(winning)
                metrics['losing_trades'] = len(losing)
                metrics['win_rate'] = (len(winning) / len(trades) * 100) if trades else 0
                
                # Calculate P&L
                metrics['gross_profit'] = sum(t['pnl'] for t in winning)
                metrics['gross_loss'] = abs(sum(t['pnl'] for t in losing))
                metrics['net_pnl'] = metrics['gross_profit'] - metrics['gross_loss']
                
                # Get top and worst trades
                sorted_trades = sorted(trades, key=lambda x: x.get('pnl', 0), reverse=True)
                metrics['top_trades'] = sorted_trades[:3]
                metrics['worst_trades'] = sorted_trades[-3:]
                
                # Calculate ratios
                if metrics['gross_loss'] > 0:
                    metrics['profit_factor'] = metrics['gross_profit'] / metrics['gross_loss']
                    
                if winning:
                    metrics['avg_win'] = metrics['gross_profit'] / len(winning)
                if losing:
                    metrics['avg_loss'] = metrics['gross_loss'] / len(losing)
                    
                if metrics['avg_loss'] > 0:
                    metrics['win_loss_ratio'] = metrics['avg_win'] / metrics['avg_loss']
            
            # Get other metrics from Redis
            metrics['error_count'] = int(self.redis.get('daily:errors:count') or 0)
            metrics['signals_generated'] = int(self.redis.get('daily:signals:count') or 0)
            
            # Calculate execution rate
            if metrics['signals_generated'] > 0:
                metrics['execution_rate'] = (metrics['total_trades'] / metrics['signals_generated'] * 100)
            
            # Get market data
            market_data = await self._get_market_data()
            metrics.update(market_data)
            
            # Get system uptime
            start_time = self.redis.get('bot:start_time')
            if start_time:
                uptime = datetime.now() - datetime.fromisoformat(start_time)
                metrics['uptime'] = str(uptime).split('.')[0]
                
        except Exception as e:
            logger.error(f"Error gathering daily metrics: {str(e)}")
            
        return metrics
    
    async def _gather_weekly_metrics(self) -> Dict[str, Any]:
        """Gather weekly performance metrics"""
        metrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'net_pnl': 0.0,
            'roi': 0.0,
            'daily_pnl': {},
            'best_day': {'date': 'N/A', 'pnl': 0, 'trades': 0},
            'worst_day': {'date': 'N/A', 'pnl': 0, 'trades': 0},
            'strategies': {},
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'btc_performance': 0.0,
            'alpha': 0.0,
            'has_chart': False
        }
        
        try:
            # Get weekly data from Redis
            for i in range(7):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                daily_data = self.redis.get(f'historical:daily:{date}')
                
                if daily_data:
                    data = json.loads(daily_data)
                    metrics['daily_pnl'][date] = data.get('pnl', 0)
                    metrics['total_trades'] += data.get('trades', 0)
                    metrics['net_pnl'] += data.get('pnl', 0)
                    
                    # Track best/worst days
                    if data.get('pnl', 0) > metrics['best_day']['pnl']:
                        metrics['best_day'] = {
                            'date': date,
                            'pnl': data['pnl'],
                            'trades': data.get('trades', 0)
                        }
                    if data.get('pnl', 0) < metrics['worst_day']['pnl']:
                        metrics['worst_day'] = {
                            'date': date,
                            'pnl': abs(data['pnl']),
                            'trades': data.get('trades', 0)
                        }
            
            # Calculate additional metrics
            # This would be more complex in real implementation
            
        except Exception as e:
            logger.error(f"Error gathering weekly metrics: {str(e)}")
            
        return metrics
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market conditions"""
        # This would connect to your market data source
        # Simplified for example
        return {
            'market_volatility': 'Normal',
            'market_trend': 'Bullish',
            'market_volume': 'Above average'
        }
    
    async def _analyze_market(self) -> Dict[str, Any]:
        """Perform market analysis"""
        # This would perform actual technical analysis
        # Simplified for example
        return {
            'trend': 'Bullish',
            'momentum': 'Strong',
            'volatility': 'Moderate',
            'volume_trend': 'Increasing',
            'rsi': 58.5,
            'macd_signal': 'Bullish',
            'support': 42000,
            'resistance': 45000,
            'fear_greed': 65,
            'funding_rate': 0.01,
            'open_interest': 2_000_000_000,
            'recommendation': 'Favor long positions on pullbacks'
        }
    
    async def _get_active_positions(self) -> List[Dict[str, Any]]:
        """Get current active positions"""
        # This would connect to your trading system
        # Simplified for example
        return []
    
    def _format_top_trades(self, trades: List[Dict[str, Any]]) -> str:
        """Format top performing trades"""
        if not trades:
            return "• No winning trades today"
            
        formatted = []
        for i, trade in enumerate(trades[:3], 1):
            formatted.append(
                f"{i}. {trade.get('symbol', 'N/A')}: "
                f"+${trade.get('pnl', 0):,.2f} "
                f"({trade.get('pnl_pct', 0):+.1f}%)"
            )
        return '\n'.join(formatted)
    
    def _format_worst_trades(self, trades: List[Dict[str, Any]]) -> str:
        """Format worst performing trades"""
        if not trades:
            return "• No losing trades today"
            
        formatted = []
        for i, trade in enumerate(trades[:3], 1):
            formatted.append(
                f"{i}. {trade.get('symbol', 'N/A')}: "
                f"-${abs(trade.get('pnl', 0)):,.2f} "
                f"({trade.get('pnl_pct', 0):.1f}%)"
            )
        return '\n'.join(formatted)
    
    def _format_strategy_performance(self, strategies: Dict[str, Any]) -> str:
        """Format strategy performance"""
        if not strategies:
            return "• No strategy data available"
            
        formatted = []
        for name, data in strategies.items():
            formatted.append(
                f"• {name}: {data.get('trades', 0)} trades, "
                f"{data.get('win_rate', 0):.1f}% win rate, "
                f"${data.get('pnl', 0):,.2f} P&L"
            )
        return '\n'.join(formatted)
    
    def _format_daily_breakdown(self, daily_pnl: Dict[str, float]) -> str:
        """Format daily P&L breakdown"""
        if not daily_pnl:
            return "• No data available"
            
        formatted = []
        for date, pnl in sorted(daily_pnl.items(), reverse=True)[:7]:
            emoji = "🟢" if pnl >= 0 else "🔴"
            formatted.append(f"• {date}: {emoji} ${pnl:,.2f}")
        return '\n'.join(formatted)
    
    def _format_weekly_strategy_analysis(self, strategies: Dict[str, Any]) -> str:
        """Format weekly strategy analysis"""
        if not strategies:
            return "• No strategy data available"
            
        formatted = []
        for name, data in strategies.items():
            formatted.append(
                f"<b>{name}:</b>\n"
                f"  • Trades: {data.get('trades', 0)}\n"
                f"  • Win Rate: {data.get('win_rate', 0):.1f}%\n"
                f"  • P&L: ${data.get('pnl', 0):,.2f}\n"
                f"  • Best Trade: ${data.get('best_trade', 0):,.2f}"
            )
        return '\n'.join(formatted)
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Generate trading recommendations based on metrics"""
        recommendations = []
        
        # Win rate analysis
        if metrics['win_rate'] < 40:
            recommendations.append("⚠️ Low win rate - review entry criteria")
        elif metrics['win_rate'] > 70:
            recommendations.append("✅ Excellent win rate - consider scaling up")
            
        # Risk/reward analysis
        if metrics['win_loss_ratio'] < 1:
            recommendations.append("⚠️ Poor risk/reward - adjust stop loss and targets")
        elif metrics['win_loss_ratio'] > 2:
            recommendations.append("✅ Great risk/reward ratio")
            
        # Drawdown analysis
        if metrics['max_drawdown'] > 10:
            recommendations.append("⚠️ High drawdown - reduce position sizes")
            
        # Execution rate
        if metrics['execution_rate'] < 50:
            recommendations.append("💡 Low execution rate - check order settings")
            
        return '\n'.join(recommendations) if recommendations else "✅ All systems operating normally"
    
    def _generate_weekly_goals(self, metrics: Dict[str, Any]) -> str:
        """Generate goals for next week based on performance"""
        goals = []
        
        if metrics['win_rate'] < 50:
            goals.append("📈 Improve win rate to 50%+")
        if metrics['roi'] < 5:
            goals.append("💰 Target 5%+ weekly ROI")
        if metrics['max_drawdown'] > 5:
            goals.append("🛡️ Keep drawdown under 5%")
            
        goals.append("🎯 Maintain consistent risk management")
        
        return '\n'.join(goals)
    
    def _get_next_summary_time(self) -> str:
        """Get next scheduled summary time"""
        current_time = datetime.now()
        for summary_time in self.summary_times:
            hour, minute = map(int, summary_time.split(':'))
            next_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if next_time > current_time:
                return next_time.strftime('%H:%M')
                
        # Next day's first summary
        return f"tomorrow at {self.summary_times[0]}"
    
    async def _reset_daily_metrics(self):
        """Reset daily metrics in Redis"""
        try:
            # Save to historical data
            date = datetime.now().strftime('%Y-%m-%d')
            daily_data = {
                'pnl': float(self.redis.get('daily:pnl') or 0),
                'trades': int(self.redis.get('daily:trades:count') or 0),
                'errors': int(self.redis.get('daily:errors:count') or 0),
                'signals': int(self.redis.get('daily:signals:count') or 0)
            }
            
            self.redis.set(f'historical:daily:{date}', json.dumps(daily_data))
            
            # Reset counters
            for key in self.metrics_keys.values():
                self.redis.delete(key)
                
            logger.info("Daily metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {str(e)}")
    
    async def _send_performance_chart(self, metrics: Dict[str, Any]):
        """Send performance chart (placeholder)"""
        # This would generate and send actual charts
        pass
    
    async def _send_weekly_chart(self, metrics: Dict[str, Any]):
        """Send weekly performance chart (placeholder)"""
        # This would generate and send actual charts
        pass
    
    async def _send_error_notification(self, component: str, error: str):
        """Send error notification"""
        await self.telegram._send_message(
            f"🚨 Error in {component}: {error}",
            parse_mode="HTML"
        )

# Convenience functions
def start_daily_summaries():
    """Start the daily summary scheduler"""
    scheduler = DailySummaryScheduler()
    scheduler.start()
    return scheduler

# Example usage
if __name__ == "__main__":
    async def test_scheduler():
        scheduler = DailySummaryScheduler()
        
        # Test immediate summary
        await scheduler._send_daily_summary()
        
        # Start scheduler
        scheduler.start()
        
        # Keep running
        try:
            await asyncio.sleep(3600)  # Run for 1 hour
        finally:
            scheduler.stop()
    
    asyncio.run(test_scheduler())
