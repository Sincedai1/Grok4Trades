#!/usr/bin/env python3
"""
Telegram Setup Utilities for Grok4Trades
Tests credentials, gets chat ID, and provides message templates
"""

import os
import sys
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TelegramSetup:
    """Utilities for setting up and testing Telegram bot"""
    
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = "https://api.telegram.org/bot{token}/{method}"
        
    async def test_credentials(self):
        """Test if the bot token is valid"""
        print("\n🔍 Testing Telegram Bot Credentials...")
        print("=" * 50)
        
        if not self.token:
            print("❌ Error: TELEGRAM_BOT_TOKEN not found in environment")
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                url = self.base_url.format(token=self.token, method="getMe")
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            bot_info = data.get("result", {})
                            print(f"✅ Bot Token Valid!")
                            print(f"   Bot Name: {bot_info.get('first_name')}")
                            print(f"   Username: @{bot_info.get('username')}")
                            print(f"   Bot ID: {bot_info.get('id')}")
                            return True
                        else:
                            print(f"❌ Error: {data.get('description', 'Unknown error')}")
                    else:
                        print(f"❌ HTTP Error: {response.status}")
                        
        except Exception as e:
            print(f"❌ Connection Error: {str(e)}")
            
        return False
        
    async def get_chat_id(self):
        """Get chat ID from recent messages"""
        print("\n📱 Getting Chat ID...")
        print("=" * 50)
        
        if not self.token:
            print("❌ Error: TELEGRAM_BOT_TOKEN not found")
            return
            
        print("Instructions:")
        print("1. Send a message to your bot on Telegram")
        print("2. If using a group, make sure the bot is added to the group")
        print("3. Wait a moment and press Enter to continue...")
        input()
        
        try:
            async with aiohttp.ClientSession() as session:
                url = self.base_url.format(token=self.token, method="getUpdates")
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok") and data.get("result"):
                            chats = {}
                            for update in data["result"]:
                                message = update.get("message", {})
                                chat = message.get("chat", {})
                                if chat:
                                    chat_id = chat.get("id")
                                    chat_type = chat.get("type")
                                    title = chat.get("title", chat.get("first_name", "Unknown"))
                                    username = chat.get("username", "")
                                    
                                    if chat_id not in chats:
                                        chats[chat_id] = {
                                            "type": chat_type,
                                            "title": title,
                                            "username": username,
                                            "last_message": message.get("text", "")
                                        }
                                        
                            if chats:
                                print("\n✅ Found the following chats:")
                                for chat_id, info in chats.items():
                                    print(f"\nChat ID: {chat_id}")
                                    print(f"  Type: {info['type']}")
                                    print(f"  Name: {info['title']}")
                                    if info['username']:
                                        print(f"  Username: @{info['username']}")
                                    print(f"  Last Message: {info['last_message'][:50]}...")
                                    
                                print(f"\n💡 Add this to your .env file:")
                                print(f"TELEGRAM_CHAT_ID={list(chats.keys())[0]}")
                            else:
                                print("❌ No messages found. Please send a message to your bot first.")
                        else:
                            print("❌ No updates found")
                            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    async def test_send_message(self):
        """Test sending a message"""
        print("\n📤 Testing Message Send...")
        print("=" * 50)
        
        if not self.token:
            print("❌ Error: TELEGRAM_BOT_TOKEN not found")
            return False
            
        if not self.chat_id:
            print("❌ Error: TELEGRAM_CHAT_ID not found")
            print("   Run option 2 first to get your chat ID")
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                url = self.base_url.format(token=self.token, method="sendMessage")
                payload = {
                    "chat_id": self.chat_id,
                    "text": "🎉 <b>Test Successful!</b>\n\nYour Telegram bot is properly configured and ready to send trading alerts.",
                    "parse_mode": "HTML"
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            print("✅ Message sent successfully!")
                            print("   Check your Telegram for the test message")
                            return True
                        else:
                            print(f"❌ Error: {data.get('description', 'Unknown error')}")
                    else:
                        print(f"❌ HTTP Error: {response.status}")
                        text = await response.text()
                        print(f"   Response: {text}")
                        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        return False
        
    def show_message_templates(self):
        """Display message templates for trading alerts"""
        print("\n📝 Message Templates")
        print("=" * 50)
        
        templates = {
            "Trade Signal": """
🚀 <b>NEW TRADE SIGNAL</b>

Symbol: <code>{symbol}</code>
Side: <b>{side}</b>
Entry: ${entry_price}
Target: ${target_price} ({profit_pct}%)
Stop Loss: ${stop_loss} ({loss_pct}%)

Risk/Reward: {risk_reward}
Confidence: {confidence}/5

<i>Strategy: {strategy_name}</i>
""",
            
            "Trade Executed": """
✅ <b>TRADE EXECUTED</b>

Symbol: <code>{symbol}</code>
Side: {side}
Amount: {amount} ({notional} USDT)
Price: ${price}
Fee: ${fee}

Order ID: <code>{order_id}</code>
Time: {timestamp}
""",
            
            "Trade Closed": """
💰 <b>TRADE CLOSED</b>

Symbol: <code>{symbol}</code>
P&L: {pnl_emoji} <b>${pnl} ({pnl_pct}%)</b>
Duration: {duration}

Entry: ${entry_price}
Exit: ${exit_price}
Size: {size}

Total Trades Today: {daily_trades}
Daily P&L: ${daily_pnl}
""",
            
            "Risk Alert": """
⚠️ <b>RISK ALERT</b>

{alert_type}

Current Drawdown: {drawdown}%
Daily Loss: ${daily_loss}
Open Positions: {open_positions}

Action: {recommended_action}
""",
            
            "System Status": """
🤖 <b>SYSTEM STATUS</b>

Status: {status_emoji} {status}
Uptime: {uptime}
Active Strategies: {active_strategies}

Account Balance: ${balance}
Free Margin: ${free_margin}
Used Margin: ${used_margin}

Last Update: {timestamp}
""",
            
            "Error Alert": """
🚨 <b>ERROR ALERT</b>

Component: {component}
Error: {error_message}
Time: {timestamp}

Action Taken: {action}
Impact: {impact}

<i>Check logs for details</i>
""",
            
            "Daily Summary": """
📊 <b>DAILY SUMMARY</b>
{date}

Total Trades: {total_trades}
Win Rate: {win_rate}%
P&L: {pnl_emoji} ${total_pnl} ({pnl_pct}%)

Best Trade: {best_trade} (+${best_pnl})
Worst Trade: {worst_trade} (-${worst_pnl})

Max Drawdown: {max_drawdown}%
Sharpe Ratio: {sharpe_ratio}

Top Performer: {top_strategy}
"""
        }
        
        print("\nAvailable message templates:\n")
        
        for name, template in templates.items():
            print(f"\n--- {name} ---")
            print(template)
            
        # Save templates to file
        with open("telegram_templates.py", "w") as f:
            f.write('"""Telegram Message Templates for Trading Bot"""\n\n')
            f.write("MESSAGE_TEMPLATES = {\n")
            for name, template in templates.items():
                f.write(f'    "{name}": """{template}""",\n\n')
            f.write("}\n\n")
            f.write("# Example usage:\n")
            f.write("# from telegram_templates import MESSAGE_TEMPLATES\n")
            f.write("# message = MESSAGE_TEMPLATES['Trade Signal'].format(\n")
            f.write("#     symbol='BTC/USDT',\n")
            f.write("#     side='BUY',\n")
            f.write("#     entry_price=45000,\n")
            f.write("#     # ... etc\n")
            f.write("# )\n")
            
        print("\n✅ Templates saved to: telegram_templates.py")
        
async def main():
    """Main menu for Telegram setup"""
    setup = TelegramSetup()
    
    while True:
        print("\n" + "=" * 50)
        print("🤖 TELEGRAM BOT SETUP FOR GROK4TRADES")
        print("=" * 50)
        print("\n1. Test Telegram credentials")
        print("2. Get Chat ID")
        print("3. Send test message")
        print("4. Show message templates")
        print("5. Run all tests")
        print("0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            await setup.test_credentials()
        elif choice == "2":
            await setup.get_chat_id()
        elif choice == "3":
            await setup.test_send_message()
        elif choice == "4":
            setup.show_message_templates()
        elif choice == "5":
            # Run all tests
            if await setup.test_credentials():
                await setup.get_chat_id()
                await setup.test_send_message()
                setup.show_message_templates()
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")
            
        if choice != "0":
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("\n🔧 Checking environment...")
    
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found")
        print("   Make sure you're in the project root directory")
        
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7+ required")
        sys.exit(1)
        
    try:
        import aiohttp
        import dotenv
    except ImportError:
        print("📦 Installing required packages...")
        os.system(f"{sys.executable} -m pip install aiohttp python-dotenv")
        print("\n✅ Packages installed. Please run the script again.")
        sys.exit(0)
        
    # Run the main menu
    asyncio.run(main())
