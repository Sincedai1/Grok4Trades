#!/usr/bin/env python3
"""
Test Telegram with quantum-sol-stack credentials
"""

import os
import sys
import asyncio
import aiohttp
from pathlib import Path

# Load environment from quantum-sol-stack/.env
def load_quantum_env():
    env_path = Path(__file__).parent / "quantum-sol-stack" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded environment from {env_path}")
    else:
        print(f"❌ Environment file not found: {env_path}")
        return False
    return True

async def test_telegram():
    """Test Telegram bot functionality"""
    
    # Load quantum-sol-stack environment
    if not load_quantum_env():
        return
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    print("\n🔍 Testing Telegram Bot")
    print("=" * 50)
    
    # Mask sensitive data for display
    if token:
        masked_token = token[:10] + "..." + token[-5:] if len(token) > 15 else "***"
        print(f"Token: {masked_token}")
    else:
        print("Token: ❌ Not found")
        return
        
    if chat_id:
        print(f"Chat ID: {chat_id}")
    else:
        print("Chat ID: ❌ Not found")
        print("\n⚠️  No TELEGRAM_CHAT_ID found. Let's get it...")
        
    # Test bot token
    print("\n🤖 Validating bot token...")
    try:
        async with aiohttp.ClientSession() as session:
            # Get bot info
            url = f"https://api.telegram.org/bot{token}/getMe"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        bot = data["result"]
                        print(f"✅ Bot validated: @{bot['username']} ({bot['first_name']})")
                        print(f"   Bot ID: {bot['id']}")
                    else:
                        print(f"❌ Invalid token: {data.get('description')}")
                        return
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    return
            
            # Get chat ID if not set
            if not chat_id:
                print("\n📱 Getting available chats...")
                print("Please send a message to your bot first, then press Enter...")
                input()
                
                url = f"https://api.telegram.org/bot{token}/getUpdates"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok") and data.get("result"):
                            chats = {}
                            for update in data["result"]:
                                message = update.get("message", {})
                                chat = message.get("chat", {})
                                if chat:
                                    cid = chat.get("id")
                                    if cid and cid not in chats:
                                        chats[cid] = {
                                            "type": chat.get("type"),
                                            "name": chat.get("title", chat.get("first_name", "Unknown")),
                                            "username": chat.get("username", ""),
                                            "text": message.get("text", "")[:50]
                                        }
                            
                            if chats:
                                print("\n✅ Found chats:")
                                for cid, info in chats.items():
                                    print(f"\nChat ID: {cid}")
                                    print(f"  Type: {info['type']}")
                                    print(f"  Name: {info['name']}")
                                    if info['username']:
                                        print(f"  Username: @{info['username']}")
                                    print(f"  Last message: {info['text']}...")
                                
                                # Use the first chat ID
                                chat_id = str(list(chats.keys())[0])
                                print(f"\n💡 Using chat ID: {chat_id}")
                            else:
                                print("❌ No messages found. Send a message to your bot first.")
                                return
            
            # Send test message
            if chat_id:
                print(f"\n📤 Sending test message to chat {chat_id}...")
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                
                # Format current time nicely
                from datetime import datetime
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                payload = {
                    "chat_id": chat_id,
                    "text": f"""🎉 <b>Telegram Integration Test Successful!</b>

✅ Bot: Connected
✅ Chat: Verified
✅ Messages: Working

🤖 <b>System Info:</b>
• Bot is ready for trading alerts
• Time: {current_time}
• Platform: Grok4Trades

<i>Your trading bot can now send you notifications!</i>""",
                    "parse_mode": "HTML"
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            print("✅ Test message sent successfully!")
                            print("   Check your Telegram!")
                            
                            # Update .env file suggestion
                            if not os.getenv("TELEGRAM_CHAT_ID"):
                                print(f"\n💡 Add this to your .env file:")
                                print(f"TELEGRAM_CHAT_ID={chat_id}")
                        else:
                            print(f"❌ Failed to send: {data.get('description')}")
                    else:
                        print(f"❌ HTTP Error: {response.status}")
                        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Grok4Trades Telegram Integration Test")
    print("=" * 50)
    
    # Check for required packages
    try:
        import aiohttp
    except ImportError:
        print("📦 Installing required packages...")
        os.system(f"{sys.executable} -m pip install aiohttp")
        print("✅ Please run the script again.")
        sys.exit(0)
    
    # Run the test
    asyncio.run(test_telegram())
