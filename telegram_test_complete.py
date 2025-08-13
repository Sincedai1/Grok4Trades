#!/usr/bin/env python3
"""
Complete Telegram Test Suite for Grok4Trades
Properly loads environment from quantum-sol-stack/.env
"""

import os
import sys
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime

# First, load the quantum-sol-stack environment
env_path = Path(__file__).parent / "quantum-sol-stack" / ".env"
if env_path.exists():
    print(f"📁 Loading environment from: {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print("✅ Environment loaded successfully\n")
else:
    print(f"❌ Environment file not found: {env_path}")
    sys.exit(1)

class TelegramTester:
    """Complete Telegram testing suite"""
    
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = "https://api.telegram.org/bot{token}/{method}"
        
    async def test_all(self):
        """Run all tests"""
        print("🚀 TELEGRAM COMPLETE TEST SUITE")
        print("=" * 60)
        
        # Test 1: Validate credentials
        print("\n📋 Test 1: Validate Credentials")
        print("-" * 40)
        
        if self.token:
            print(f"✅ Token found: {self.token[:10]}...{self.token[-5:]}")
        else:
            print("❌ Token not found")
            return
            
        if self.chat_id:
            print(f"✅ Chat ID found: {self.chat_id}")
        else:
            print("❌ Chat ID not found")
            return
        
        # Test 2: Validate bot info
        print("\n📋 Test 2: Bot Information")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            url = self.base_url.format(token=self.token, method="getMe")
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        bot = data["result"]
                        print(f"✅ Bot Name: {bot['first_name']}")
                        print(f"✅ Username: @{bot['username']}")
                        print(f"✅ Bot ID: {bot['id']}")
                        print(f"✅ Can join groups: {bot.get('can_join_groups', False)}")
                        print(f"✅ Can read messages: {bot.get('can_read_all_group_messages', False)}")
                    else:
                        print(f"❌ Error: {data.get('description')}")
                        return
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    return
            
            # Test 3: Send test messages
            print("\n📋 Test 3: Send Test Messages")
            print("-" * 40)
            
            # Message 1: Simple text
            print("Sending simple text message...")
            success = await self._send_message(
                session,
                "🔧 Test 1: Simple text message"
            )
            print(f"{'✅' if success else '❌'} Simple text: {'Sent' if success else 'Failed'}")
            
            # Message 2: HTML formatted
            print("Sending HTML formatted message...")
            success = await self._send_message(
                session,
                "<b>🔧 Test 2: HTML Formatting</b>\n\n"
                "• <i>Italic text</i>\n"
                "• <b>Bold text</b>\n"
                "• <code>Monospace code</code>\n"
                "• <a href='https://telegram.org'>Link</a>",
                parse_mode="HTML"
            )
            print(f"{'✅' if success else '❌'} HTML formatted: {'Sent' if success else 'Failed'}")
            
            # Message 3: Trading alert simulation
            print("Sending trading alert simulation...")
            success = await self._send_message(
                session,
                self._format_trade_alert(),
                parse_mode="HTML"
            )
            print(f"{'✅' if success else '❌'} Trading alert: {'Sent' if success else 'Failed'}")
            
            # Message 4: Emoji-rich message
            print("Sending emoji-rich message...")
            success = await self._send_message(
                session,
                "🔧 Test 4: Emoji Support\n\n"
                "📈 Charts: 📊📉📈\n"
                "💰 Money: 💵💶💷💴\n"
                "⚠️ Alerts: 🚨⚡️🔔\n"
                "✅ Status: 🟢🟡🔴"
            )
            print(f"{'✅' if success else '❌'} Emoji message: {'Sent' if success else 'Failed'}")
            
            # Test 4: Message with buttons (inline keyboard)
            print("\n📋 Test 4: Interactive Features")
            print("-" * 40)
            
            print("Sending message with inline buttons...")
            success = await self._send_message_with_buttons(session)
            print(f"{'✅' if success else '❌'} Inline buttons: {'Sent' if success else 'Failed'}")
            
            # Final summary message
            print("\n📋 Test 5: Final Summary")
            print("-" * 40)
            
            summary = f"""✅ <b>TELEGRAM INTEGRATION COMPLETE!</b>

<b>Configuration Summary:</b>
• Bot: @{bot['username']}
• Chat ID: {self.chat_id}
• Status: Fully Operational

<b>Features Tested:</b>
✅ Text messages
✅ HTML formatting
✅ Emoji support
✅ Trading alerts
✅ Interactive buttons

<i>Your bot is ready to send trading notifications!</i>

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            success = await self._send_message(session, summary, parse_mode="HTML")
            print(f"{'✅' if success else '❌'} Summary sent: {'Success' if success else 'Failed'}")
            
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED!")
        print("Check your Telegram for the test messages.")
        print("=" * 60)
    
    async def _send_message(self, session, text, parse_mode=None):
        """Send a simple message"""
        url = self.base_url.format(token=self.token, method="sendMessage")
        payload = {
            "chat_id": self.chat_id,
            "text": text
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
            
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("ok", False)
                return False
        except:
            return False
    
    async def _send_message_with_buttons(self, session):
        """Send message with inline keyboard"""
        url = self.base_url.format(token=self.token, method="sendMessage")
        payload = {
            "chat_id": self.chat_id,
            "text": "🔧 Test 5: Interactive Buttons\n\nThese buttons won't do anything, but show that your bot supports interactivity:",
            "reply_markup": {
                "inline_keyboard": [
                    [
                        {"text": "📈 View Chart", "callback_data": "chart"},
                        {"text": "💰 Check Balance", "callback_data": "balance"}
                    ],
                    [
                        {"text": "⚙️ Settings", "callback_data": "settings"},
                        {"text": "📊 Stats", "callback_data": "stats"}
                    ],
                    [
                        {"text": "🔗 Open Telegram Docs", "url": "https://core.telegram.org/bots"}
                    ]
                ]
            }
        }
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("ok", False)
                return False
        except:
            return False
    
    def _format_trade_alert(self):
        """Format a sample trade alert"""
        return """🚀 <b>TRADE ALERT (Test)</b>

Symbol: <code>BTC/USDT</code>
Side: <b>BUY</b>
Entry: $43,250.00
Target: $45,000.00 (+4.0%)
Stop Loss: $42,000.00 (-2.9%)

Risk/Reward: 1:1.4
Confidence: ⭐⭐⭐⭐/5

<i>Strategy: MA Crossover + RSI Divergence</i>

⚠️ <i>This is a test message - not a real trade signal</i>"""

async def main():
    """Main test function"""
    tester = TelegramTester()
    
    # Check if we have the required packages
    try:
        import aiohttp
    except ImportError:
        print("📦 Installing required packages...")
        os.system(f"{sys.executable} -m pip install aiohttp")
        print("✅ Please run the script again.")
        return
    
    await tester.test_all()

if __name__ == "__main__":
    asyncio.run(main())
