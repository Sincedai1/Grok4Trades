"""
Custom Alert Types for Grok4Trades
Extended notification templates and alert functions
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json

from telegram_notifier import TelegramNotifier, AlertType

class ExtendedAlertType(Enum):
    """Extended alert types for trading bot"""
    MARKET_ANALYSIS = "market_analysis"
    VOLUME_SPIKE = "volume_spike"
    TREND_CHANGE = "trend_change"
    CORRELATION_ALERT = "correlation_alert"
    NEWS_IMPACT = "news_impact"
    WHALE_ACTIVITY = "whale_activity"
    FUNDING_RATE = "funding_rate"
    LIQUIDATION_ALERT = "liquidation"
    STRATEGY_PERFORMANCE = "strategy_perf"
    ACCOUNT_MILESTONE = "milestone"

class CustomAlerts:
    """Extended alert functionality for trading bot"""
    
    def __init__(self, notifier: Optional[TelegramNotifier] = None):
        self.notifier = notifier or TelegramNotifier()
        
    async def send_market_analysis(self, analysis_data: Dict[str, Any]):
        """Send comprehensive market analysis"""
        message = f"""📊 <b>MARKET ANALYSIS</b>
{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

<b>Market Overview:</b>
• Trend: {analysis_data.get('trend', 'Neutral')} {self._get_trend_emoji(analysis_data.get('trend'))}
• Momentum: {analysis_data.get('momentum', 'Neutral')}
• Volatility: {analysis_data.get('volatility', 'Normal')}
• Volume: {analysis_data.get('volume_trend', 'Average')}

<b>Technical Indicators:</b>
• RSI: {analysis_data.get('rsi', 50):.1f} {self._get_rsi_status(analysis_data.get('rsi', 50))}
• MACD: {analysis_data.get('macd_signal', 'Neutral')}
• Support: ${analysis_data.get('support', 0):,.2f}
• Resistance: ${analysis_data.get('resistance', 0):,.2f}

<b>Market Sentiment:</b>
• Fear & Greed: {analysis_data.get('fear_greed', 50)}/100
• Funding Rate: {analysis_data.get('funding_rate', 0):.4f}%
• Open Interest: ${analysis_data.get('open_interest', 0):,.0f}

<b>Recommendation:</b>
{analysis_data.get('recommendation', 'Monitor closely')}"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    async def send_volume_spike_alert(self, symbol: str, volume_data: Dict[str, Any]):
        """Alert for unusual volume spikes"""
        spike_pct = volume_data.get('spike_percentage', 0)
        emoji = "🚀" if spike_pct > 200 else "📈" if spike_pct > 100 else "📊"
        
        message = f"""{emoji} <b>VOLUME SPIKE DETECTED</b>

Symbol: <code>{symbol}</code>
Volume Increase: <b>{spike_pct:.1f}%</b>
Current Volume: {volume_data.get('current_volume', 0):,.0f}
Average Volume: {volume_data.get('avg_volume', 0):,.0f}

Price Action:
• Current: ${volume_data.get('current_price', 0):,.2f}
• Change: {volume_data.get('price_change_pct', 0):+.2f}%

<b>Interpretation:</b>
{self._interpret_volume_spike(spike_pct, volume_data.get('price_change_pct', 0))}

⚠️ <i>High volume often precedes significant moves</i>"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    async def send_trend_change_alert(self, symbol: str, trend_data: Dict[str, Any]):
        """Alert for trend changes"""
        old_trend = trend_data.get('old_trend', 'Unknown')
        new_trend = trend_data.get('new_trend', 'Unknown')
        
        message = f"""🔄 <b>TREND CHANGE ALERT</b>

Symbol: <code>{symbol}</code>

Trend Change: {old_trend} ➡️ <b>{new_trend}</b>

<b>Confirmation Signals:</b>
• MA Cross: {trend_data.get('ma_confirmation', 'No')}
• Volume: {trend_data.get('volume_confirmation', 'No')}
• Momentum: {trend_data.get('momentum_confirmation', 'No')}

<b>Key Levels:</b>
• Entry Zone: ${trend_data.get('entry_zone_low', 0):,.2f} - ${trend_data.get('entry_zone_high', 0):,.2f}
• Stop Loss: ${trend_data.get('suggested_stop', 0):,.2f}
• Target 1: ${trend_data.get('target_1', 0):,.2f}
• Target 2: ${trend_data.get('target_2', 0):,.2f}

Confidence: {'⭐' * trend_data.get('confidence', 3)}/5

<i>Consider adjusting positions based on new trend</i>"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    async def send_correlation_alert(self, pairs: List[Dict[str, Any]]):
        """Alert for correlation opportunities"""
        message = """🔗 <b>CORRELATION ALERT</b>

<b>Detected Correlations:</b>
"""
        for pair in pairs[:5]:  # Limit to 5 pairs
            message += f"\n• {pair['pair1']} ↔️ {pair['pair2']}"
            message += f"\n  Correlation: {pair['correlation']:.2f}"
            message += f"\n  Type: {pair['type']}\n"
        
        message += f"""
<b>Trading Opportunity:</b>
{pairs[0].get('opportunity', 'Monitor for divergence')}

<i>Use correlations for pair trading or hedging</i>"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    async def send_whale_activity_alert(self, activity_data: Dict[str, Any]):
        """Alert for large transactions or whale movements"""
        message = f"""🐋 <b>WHALE ACTIVITY DETECTED</b>

Transaction Type: {activity_data.get('type', 'Unknown')}
Amount: {activity_data.get('amount', 0):,.2f} {activity_data.get('currency', 'BTC')}
Value: ${activity_data.get('usd_value', 0):,.0f}

<b>Details:</b>
• From: {self._truncate_address(activity_data.get('from_address', ''))}
• To: {self._truncate_address(activity_data.get('to_address', ''))}
• Exchange: {activity_data.get('exchange', 'Unknown')}

<b>Market Impact:</b>
{activity_data.get('expected_impact', 'Monitor for price movement')}

<i>Large transfers may indicate upcoming volatility</i>"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    async def send_funding_rate_alert(self, funding_data: Dict[str, Any]):
        """Alert for significant funding rate changes"""
        rate = funding_data.get('current_rate', 0)
        emoji = "🔴" if abs(rate) > 0.05 else "🟡" if abs(rate) > 0.02 else "🟢"
        
        message = f"""{emoji} <b>FUNDING RATE ALERT</b>

Symbol: <code>{funding_data.get('symbol', 'BTC/USDT')}</code>
Current Rate: <b>{rate:.4f}%</b>
Next Payment: {funding_data.get('next_payment', 'In 1 hour')}

<b>Historical Context:</b>
• 24h Average: {funding_data.get('avg_24h', 0):.4f}%
• 7d Average: {funding_data.get('avg_7d', 0):.4f}%

<b>Market Sentiment:</b>
{self._interpret_funding_rate(rate)}

<b>Strategy:</b>
{self._funding_rate_strategy(rate)}"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    async def send_liquidation_alert(self, liquidation_data: Dict[str, Any]):
        """Alert for major liquidation events"""
        total_liquidated = liquidation_data.get('total_usd', 0)
        
        message = f"""💥 <b>LIQUIDATION CASCADE</b>

Total Liquidated: <b>${total_liquidated:,.0f}</b>

<b>Breakdown:</b>
• Long Liquidations: ${liquidation_data.get('long_liq', 0):,.0f}
• Short Liquidations: ${liquidation_data.get('short_liq', 0):,.0f}
• Ratio: {liquidation_data.get('long_short_ratio', 1):.2f}

<b>Largest Single Liquidation:</b>
• Size: ${liquidation_data.get('largest_size', 0):,.0f}
• Price: ${liquidation_data.get('largest_price', 0):,.2f}

<b>Market Impact:</b>
{self._assess_liquidation_impact(total_liquidated, liquidation_data.get('long_short_ratio', 1))}

⚠️ <i>High liquidations may cause volatile price swings</i>"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    async def send_strategy_performance_update(self, performance_data: Dict[str, Any]):
        """Send strategy performance update"""
        message = f"""📈 <b>STRATEGY PERFORMANCE UPDATE</b>

Strategy: {performance_data.get('strategy_name', 'Unknown')}
Period: {performance_data.get('period', 'Last 24h')}

<b>Performance Metrics:</b>
• Total Trades: {performance_data.get('total_trades', 0)}
• Win Rate: {performance_data.get('win_rate', 0):.1f}%
• Profit Factor: {performance_data.get('profit_factor', 1.0):.2f}
• Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.2f}

<b>P&L Summary:</b>
• Gross Profit: ${performance_data.get('gross_profit', 0):,.2f}
• Gross Loss: ${performance_data.get('gross_loss', 0):,.2f}
• Net P&L: <b>${performance_data.get('net_pnl', 0):,.2f}</b>
• ROI: {performance_data.get('roi', 0):.2f}%

<b>Best Trade:</b>
• Symbol: {performance_data.get('best_trade_symbol', 'N/A')}
• P&L: +${performance_data.get('best_trade_pnl', 0):,.2f}

<b>Worst Trade:</b>
• Symbol: {performance_data.get('worst_trade_symbol', 'N/A')}
• P&L: -${performance_data.get('worst_trade_pnl', 0):,.2f}

<b>Recommendation:</b>
{self._strategy_recommendation(performance_data)}"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    async def send_account_milestone(self, milestone_data: Dict[str, Any]):
        """Send account milestone notifications"""
        milestone_type = milestone_data.get('type', 'balance')
        
        emojis = {
            'balance': '💰',
            'trades': '📊',
            'profit': '🎯',
            'streak': '🔥'
        }
        
        emoji = emojis.get(milestone_type, '🎉')
        
        message = f"""{emoji} <b>MILESTONE ACHIEVED!</b>

🎊 {milestone_data.get('achievement', 'New milestone reached')}

<b>Details:</b>
• Type: {milestone_type.title()}
• Value: {milestone_data.get('value', 'N/A')}
• Previous: {milestone_data.get('previous', 'N/A')}

<b>Account Stats:</b>
• Total Balance: ${milestone_data.get('balance', 0):,.2f}
• All-Time P&L: ${milestone_data.get('all_time_pnl', 0):,.2f}
• Total Trades: {milestone_data.get('total_trades', 0)}
• Active Since: {milestone_data.get('start_date', 'Unknown')}

Keep up the great work! 🚀"""
        
        return await self.notifier._send_message(message, parse_mode="HTML")
    
    # Helper methods
    def _get_trend_emoji(self, trend: str) -> str:
        """Get emoji for trend direction"""
        trend_lower = trend.lower()
        if 'bull' in trend_lower or 'up' in trend_lower:
            return "📈"
        elif 'bear' in trend_lower or 'down' in trend_lower:
            return "📉"
        else:
            return "➡️"
    
    def _get_rsi_status(self, rsi: float) -> str:
        """Get RSI status description"""
        if rsi >= 70:
            return "⚠️ Overbought"
        elif rsi <= 30:
            return "⚠️ Oversold"
        else:
            return "✅ Neutral"
    
    def _interpret_volume_spike(self, spike_pct: float, price_change: float) -> str:
        """Interpret volume spike meaning"""
        if spike_pct > 300:
            if price_change > 2:
                return "Strong bullish momentum - potential breakout"
            elif price_change < -2:
                return "Strong bearish momentum - potential breakdown"
            else:
                return "Major accumulation/distribution - watch for direction"
        elif spike_pct > 150:
            if abs(price_change) > 1:
                return "Increased activity confirming price movement"
            else:
                return "Building pressure - breakout possible"
        else:
            return "Moderate increase in trading activity"
    
    def _truncate_address(self, address: str) -> str:
        """Truncate crypto address for display"""
        if len(address) > 12:
            return f"{address[:6]}...{address[-4:]}"
        return address
    
    def _interpret_funding_rate(self, rate: float) -> str:
        """Interpret funding rate meaning"""
        if rate > 0.05:
            return "🔴 Extremely bullish sentiment - longs paying high premium"
        elif rate > 0.02:
            return "🟡 Bullish sentiment - consider contrarian short"
        elif rate < -0.02:
            return "🟡 Bearish sentiment - consider contrarian long"
        elif rate < -0.05:
            return "🔴 Extremely bearish sentiment - shorts paying high premium"
        else:
            return "🟢 Neutral sentiment - balanced market"
    
    def _funding_rate_strategy(self, rate: float) -> str:
        """Suggest strategy based on funding rate"""
        if abs(rate) > 0.05:
            return "Consider delta-neutral arbitrage or contrarian positions"
        elif abs(rate) > 0.02:
            return "Monitor for potential reversal opportunities"
        else:
            return "Normal conditions - follow primary strategy"
    
    def _assess_liquidation_impact(self, total_usd: float, ratio: float) -> str:
        """Assess market impact of liquidations"""
        if total_usd > 100_000_000:
            if ratio > 2:
                return "🔴 Massive long squeeze - expect sharp downward pressure"
            elif ratio < 0.5:
                return "🔴 Massive short squeeze - expect sharp upward pressure"
            else:
                return "🔴 Major liquidation event - extreme volatility expected"
        elif total_usd > 50_000_000:
            return "🟡 Significant liquidations - increased volatility likely"
        else:
            return "🟢 Moderate liquidations - normal market conditions"
    
    def _strategy_recommendation(self, performance: Dict[str, Any]) -> str:
        """Generate strategy recommendation based on performance"""
        win_rate = performance.get('win_rate', 50)
        profit_factor = performance.get('profit_factor', 1.0)
        
        if win_rate > 60 and profit_factor > 1.5:
            return "✅ Excellent performance - consider increasing position sizes"
        elif win_rate > 50 and profit_factor > 1.2:
            return "✅ Good performance - maintain current settings"
        elif win_rate < 40 or profit_factor < 0.8:
            return "⚠️ Poor performance - review strategy parameters"
        else:
            return "🟡 Average performance - monitor and optimize"

# Convenience functions
custom_alerts = CustomAlerts()

async def send_custom_alert(alert_type: ExtendedAlertType, data: Dict[str, Any]):
    """Send a custom alert based on type"""
    if alert_type == ExtendedAlertType.MARKET_ANALYSIS:
        return await custom_alerts.send_market_analysis(data)
    elif alert_type == ExtendedAlertType.VOLUME_SPIKE:
        return await custom_alerts.send_volume_spike_alert(data.get('symbol', 'BTC/USDT'), data)
    elif alert_type == ExtendedAlertType.TREND_CHANGE:
        return await custom_alerts.send_trend_change_alert(data.get('symbol', 'BTC/USDT'), data)
    elif alert_type == ExtendedAlertType.CORRELATION_ALERT:
        return await custom_alerts.send_correlation_alert(data.get('pairs', []))
    elif alert_type == ExtendedAlertType.WHALE_ACTIVITY:
        return await custom_alerts.send_whale_activity_alert(data)
    elif alert_type == ExtendedAlertType.FUNDING_RATE:
        return await custom_alerts.send_funding_rate_alert(data)
    elif alert_type == ExtendedAlertType.LIQUIDATION_ALERT:
        return await custom_alerts.send_liquidation_alert(data)
    elif alert_type == ExtendedAlertType.STRATEGY_PERFORMANCE:
        return await custom_alerts.send_strategy_performance_update(data)
    elif alert_type == ExtendedAlertType.ACCOUNT_MILESTONE:
        return await custom_alerts.send_account_milestone(data)

# Example usage
if __name__ == "__main__":
    async def test_custom_alerts():
        # Test market analysis
        await custom_alerts.send_market_analysis({
            'trend': 'Bullish',
            'momentum': 'Strong',
            'volatility': 'High',
            'volume_trend': 'Increasing',
            'rsi': 65.5,
            'macd_signal': 'Bullish Cross',
            'support': 42000,
            'resistance': 45000,
            'fear_greed': 75,
            'funding_rate': 0.0125,
            'open_interest': 2_500_000_000,
            'recommendation': 'Look for pullback entries in the 43,000-43,500 range'
        })
        
        # Test volume spike
        await custom_alerts.send_volume_spike_alert('BTC/USDT', {
            'spike_percentage': 250,
            'current_volume': 5_000_000,
            'avg_volume': 2_000_000,
            'current_price': 44_250,
            'price_change_pct': 3.5
        })
    
    asyncio.run(test_custom_alerts())
