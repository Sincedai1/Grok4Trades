#!/usr/bin/env python3
"""
Get Telegram Chat ID
"""

import os
import asyncio
import aiohttp
from pathlib import Path

# Load environment from quantum-sol-stack/.env
env_path = Path(__file__).parent / "quantum-sol-stack" / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Only load the bot token, not the placeholder chat ID
                if key.strip() == "TELEGRAM_BOT_TOKEN":
                    os.environ[key.strip()] = value.strip()

async def get_chat_id():
    """Get chat ID from Telegram updates"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found")
        return
    
    print("🤖 Getting Chat ID for @Sincedai1_bot")
    print("=" * 50)
    print("\n📱 Instructions:")
    print("1. Open Telegram")
    print("2. Search for @Sincedai1_bot")
    print("3. Start a conversation with /start")
    print("4. Send a message like 'Hello Bot'")
    print("\nAfter sending a message, press Enter to continue...")
    input()
    
    print("\n🔍 Checking for messages...")
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok") and data.get("result"):
                        print(f"\n✅ Found {len(data['result'])} updates\n")
                        
                        chats_seen = {}
                        for update in data["result"]:
                            message = update.get("message", {})
                            chat = message.get("chat", {})
                            
                            if chat and chat.get("id"):
                                chat_id = chat["id"]
                                if chat_id not in chats_seen:
                                    chat_type = chat.get("type", "unknown")
                                    chat_name = chat.get("title") or chat.get("first_name") or "Unknown"
                                    username = chat.get("username", "")
                                    text = message.get("text", "")
                                    
                                    print(f"Chat Found:")
                                    print(f"  Chat ID: {chat_id}")
                                    print(f"  Type: {chat_type}")
                                    print(f"  Name: {chat_name}")
                                    if username:
                                        print(f"  Username: @{username}")
                                    print(f"  Last message: {text}")
                                    print("-" * 40)
                                    
                                    chats_seen[chat_id] = {
                                        "name": chat_name,
                                        "type": chat_type
                                    }
                        
                        if chats_seen:
                            print(f"\n💡 To use this bot, update your .env file:")
                            print(f"\nIn quantum-sol-stack/.env, replace:")
                            print(f"TELEGRAM_CHAT_ID=your_telegram_chat_id")
                            print(f"\nWith:")
                            print(f"TELEGRAM_CHAT_ID={list(chats_seen.keys())[0]}")
                            
                            # Test sending a message
                            chat_id = list(chats_seen.keys())[0]
                            print(f"\n📤 Sending test message to chat {chat_id}...")
                            
                            send_url = f"https://api.telegram.org/bot{token}/sendMessage"
                            payload = {
                                "chat_id": chat_id,
                                "text": "✅ Chat ID verified! Update your .env file with this chat ID.",
                                "parse_mode": "HTML"
                            }
                            
                            async with session.post(send_url, json=payload) as resp:
                                if resp.status == 200:
                                    result = await resp.json()
                                    if result.get("ok"):
                                        print("✅ Test message sent! Check your Telegram.")
                                    else:
                                        print(f"❌ Failed: {result.get('description')}")
                                else:
                                    print(f"❌ HTTP Error: {resp.status}")
                        else:
                            print("❌ No chats found. Make sure you sent a message to the bot.")
                    else:
                        print("❌ No updates found. Make sure you sent a message to @Sincedai1_bot")
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    text = await response.text()
                    print(f"Response: {text}")
                    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(get_chat_id())
