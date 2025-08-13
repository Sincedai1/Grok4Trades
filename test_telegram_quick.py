#!/usr/bin/env python3
"""
Quick Telegram Test Script
Tests your current Telegram credentials without menu
"""

import os
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def quick_test():
    """Quick test of Telegram credentials"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    print("🔍 Quick Telegram Test")
    print("=" * 40)
    
    # Check environment variables
    print(f"Token found: {'✅' if token else '❌'}")
    print(f"Chat ID found: {'✅' if chat_id else '❌'}")
    
    if not token:
        print("\n❌ TELEGRAM_BOT_TOKEN not found in .env")
        return
        
    # Test bot token
    print("\n🤖 Testing bot token...")
    try:
        async with aiohttp.ClientSession() as session:
            # Get bot info
            url = f"https://api.telegram.org/bot{token}/getMe"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        bot = data["result"]
                        print(f"✅ Bot: @{bot['username']} ({bot['first_name']})")
                    else:
                        print(f"❌ Error: {data.get('description')}")
                        return
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    return
                    
            # Test sending message if chat_id exists
            if chat_id:
                print(f"\n📤 Sending test message to chat {chat_id}...")
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": "✅ <b>Telegram Test Successful!</b>\n\nYour bot is configured correctly.",
                    "parse_mode": "HTML"
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            print("✅ Message sent successfully!")
                        else:
                            print(f"❌ Send failed: {data.get('description')}")
                    else:
                        print(f"❌ HTTP Error: {response.status}")
            else:
                print("\n⚠️  No TELEGRAM_CHAT_ID found")
                print("   Run telegram_setup.py to get your chat ID")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(quick_test())
