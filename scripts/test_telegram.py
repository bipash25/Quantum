#!/usr/bin/env python3
"""
Quantum Trading AI - Telegram Bot Test
=======================================
Quick test script to verify Telegram bot configuration.
"""
import asyncio
import os
import sys

# Load environment
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")


async def test_telegram():
    """Test Telegram bot connectivity"""
    import aiohttp

    print("=" * 60)
    print("Quantum Trading AI - Telegram Bot Test")
    print("=" * 60)

    if not BOT_TOKEN or BOT_TOKEN.startswith("PLACEHOLDER"):
        print("ERROR: TELEGRAM_BOT_TOKEN not configured in .env")
        return False

    if not CHANNEL_ID or CHANNEL_ID.startswith("PLACEHOLDER"):
        print("ERROR: TELEGRAM_CHANNEL_ID not configured in .env")
        return False

    print(f"Bot Token: {BOT_TOKEN[:20]}...{BOT_TOKEN[-10:]}")
    print(f"Channel ID: {CHANNEL_ID}")
    print()

    base_url = f"https://api.telegram.org/bot{BOT_TOKEN}"

    async with aiohttp.ClientSession() as session:
        # Test 1: Get bot info
        print("1. Testing getMe...")
        async with session.get(f"{base_url}/getMe") as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("ok"):
                    bot_info = data["result"]
                    print(f"   Bot Name: {bot_info.get('first_name')}")
                    print(f"   Bot Username: @{bot_info.get('username')}")
                    print(f"   Can Join Groups: {bot_info.get('can_join_groups')}")
                    print("   Status: OK")
                else:
                    print(f"   ERROR: {data.get('description')}")
                    return False
            else:
                print(f"   ERROR: HTTP {resp.status}")
                return False

        # Test 2: Send test message to channel
        print()
        print("2. Testing sendMessage to channel...")
        test_message = """
*Quantum Trading AI - Test Message*

This is a test message to verify the bot configuration.

If you see this message, the bot is correctly configured!

_This message will be deleted._
"""

        payload = {
            "chat_id": CHANNEL_ID,
            "text": test_message,
            "parse_mode": "Markdown",
        }

        async with session.post(f"{base_url}/sendMessage", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("ok"):
                    message_id = data["result"]["message_id"]
                    print(f"   Message sent successfully!")
                    print(f"   Message ID: {message_id}")
                    print("   Status: OK")

                    # Delete the test message
                    print()
                    print("3. Cleaning up (deleting test message)...")
                    delete_payload = {
                        "chat_id": CHANNEL_ID,
                        "message_id": message_id,
                    }
                    async with session.post(f"{base_url}/deleteMessage", json=delete_payload) as del_resp:
                        if del_resp.status == 200:
                            print("   Test message deleted")
                        else:
                            print("   Could not delete test message (may need admin rights)")
                else:
                    error = data.get("description", "Unknown error")
                    print(f"   ERROR: {error}")
                    if "chat not found" in error.lower():
                        print("   HINT: Make sure the bot is added as an admin to the channel")
                    return False
            else:
                error_data = await resp.json()
                print(f"   ERROR: HTTP {resp.status} - {error_data.get('description')}")
                return False

    print()
    print("=" * 60)
    print("All Telegram tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        # Install required packages if not present
        try:
            import aiohttp
            from dotenv import load_dotenv
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "python-dotenv", "-q"])
            import aiohttp
            from dotenv import load_dotenv

        result = asyncio.run(test_telegram())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
