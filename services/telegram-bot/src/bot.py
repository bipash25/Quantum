"""
Quantum Trading AI - Telegram Bot
=================================
Sends trading signals to Telegram channel and handles user commands.

Features:
- Subscribe to Redis for real-time signal updates
- Format signals as beautiful Telegram messages
- Handle user commands (/start, /signals, /status, etc.)
- Track signal delivery for analytics
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any

import aiohttp
import redis.asyncio as redis

from .config import settings


logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for signal delivery"""

    BASE_URL = "https://api.telegram.org/bot"

    def __init__(self):
        self.token = settings.telegram_token
        self.channel_id = settings.telegram_channel_id
        self.api_url = f"{self.BASE_URL}{self.token}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.redis: Optional[redis.Redis] = None

    async def start(self):
        """Initialize bot connections"""
        self.session = aiohttp.ClientSession()
        self.redis = redis.from_url(settings.redis_url)

        # Test connection
        me = await self.get_me()
        if me:
            logger.info(f"Bot started: @{me.get('username', 'unknown')}")
        else:
            logger.error("Failed to connect to Telegram")

    async def stop(self):
        """Close connections"""
        if self.session:
            await self.session.close()
        if self.redis:
            await self.redis.close()

    async def get_me(self) -> Optional[Dict]:
        """Get bot info"""
        try:
            async with self.session.get(f"{self.api_url}/getMe") as resp:
                data = await resp.json()
                if data.get("ok"):
                    return data.get("result")
        except Exception as e:
            logger.error(f"Failed to get bot info: {e}")
        return None

    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "Markdown",
        disable_preview: bool = True
    ) -> bool:
        """Send message to chat/channel"""
        try:
            async with self.session.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": disable_preview,
                }
            ) as resp:
                data = await resp.json()
                if data.get("ok"):
                    logger.info(f"Message sent to {chat_id}")
                    return True
                else:
                    logger.error(f"Failed to send message: {data}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
        return False

    def format_signal(self, signal: Dict) -> str:
        """Format signal as Telegram message"""
        direction = signal.get("direction", "UNKNOWN")
        symbol = signal.get("symbol", "???")
        emoji = "🟢" if direction == "LONG" else "🔴"

        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        tp1 = signal.get("take_profit_1", 0)
        tp2 = signal.get("take_profit_2", 0)
        tp3 = signal.get("take_profit_3", 0)
        confidence = signal.get("confidence", 0)
        rr = signal.get("risk_reward", 0)
        timeframe = signal.get("timeframe", "4h")
        reasoning = signal.get("reasoning", "Technical analysis")
        valid_until = signal.get("valid_until", "")

        # Calculate percentages
        if direction == "LONG":
            sl_pct = abs((entry - sl) / entry * 100) if entry > 0 else 0
            tp1_pct = abs((tp1 - entry) / entry * 100) if entry > 0 else 0
            tp2_pct = abs((tp2 - entry) / entry * 100) if entry > 0 else 0
            tp3_pct = abs((tp3 - entry) / entry * 100) if entry > 0 else 0
        else:
            sl_pct = abs((sl - entry) / entry * 100) if entry > 0 else 0
            tp1_pct = abs((entry - tp1) / entry * 100) if entry > 0 else 0
            tp2_pct = abs((entry - tp2) / entry * 100) if entry > 0 else 0
            tp3_pct = abs((entry - tp3) / entry * 100) if entry > 0 else 0

        # Format valid until time
        if valid_until:
            try:
                dt = datetime.fromisoformat(valid_until.replace('Z', '+00:00'))
                valid_str = dt.strftime("%H:%M UTC")
            except:
                valid_str = valid_until[:16]
        else:
            valid_str = "N/A"

        # Confidence indicator
        if confidence >= 70:
            conf_indicator = "🔥"
        elif confidence >= 60:
            conf_indicator = "⚡"
        else:
            conf_indicator = "📊"

        message = f"""
{emoji} *{direction} Signal* | {symbol} {conf_indicator}

📊 *Timeframe:* {timeframe.upper()}
💰 *Entry:* `${entry:.4f}`
🎯 *Take Profits:*
   TP1: `${tp1:.4f}` (+{tp1_pct:.1f}%)
   TP2: `${tp2:.4f}` (+{tp2_pct:.1f}%)
   TP3: `${tp3:.4f}` (+{tp3_pct:.1f}%)
🛑 *Stop Loss:* `${sl:.4f}` (-{sl_pct:.1f}%)
📏 *R:R Ratio:* {rr:.1f}:1
🎯 *Confidence:* {confidence:.0f}%
⏰ *Valid Until:* {valid_str}

💡 *Analysis:* {reasoning}

⚠️ *RISK WARNING:* Not financial advice. You can lose money trading. Past performance does not guarantee future results. Trade at your own risk.

_📈 @QuantumTradingAIX_
"""
        return message.strip()

    async def send_signal_to_channel(self, signal: Dict) -> bool:
        """Send formatted signal to the channel"""
        message = self.format_signal(signal)
        return await self.send_message(self.channel_id, message)

    async def subscribe_to_signals(self):
        """Subscribe to Redis channel for signals"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("quantum:signals")

        logger.info("Subscribed to quantum:signals channel")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    signal = json.loads(data)

                    logger.info(f"Received signal: {signal.get('symbol')} {signal.get('direction')}")
                    await self.send_signal_to_channel(signal)

                except Exception as e:
                    logger.error(f"Error processing signal: {e}")

    async def send_startup_message(self):
        """Send startup notification with disclaimer"""
        message = """
🚀 *Quantum Trading AI* is now online!

📊 Monitoring 16 crypto pairs for trading opportunities
⏱ Timeframe: 4H
🎯 Strategy: ML-powered technical analysis

Stay tuned for signals! 📈

⚠️ *RISK WARNING:* Trading signals are for educational purposes only. Not financial advice. You can lose money trading. Past performance does not guarantee future results. Trade at your own risk.

_By using this service, you accept our Terms of Service._
_Type /help in the bot for commands_
"""
        await self.send_message(self.channel_id, message.strip())

    async def run(self):
        """Main run loop"""
        await self.start()
        await self.send_startup_message()

        # Start polling for user commands in background
        asyncio.create_task(self.poll_updates())

        try:
            await self.subscribe_to_signals()
        except asyncio.CancelledError:
            logger.info("Bot shutting down...")
        finally:
            await self.stop()

    async def poll_updates(self):
        """Poll for incoming messages (user commands)"""
        offset = 0
        while True:
            try:
                async with self.session.get(
                    f"{self.api_url}/getUpdates",
                    params={"offset": offset, "timeout": 30}
                ) as resp:
                    data = await resp.json()
                    if data.get("ok"):
                        for update in data.get("result", []):
                            offset = update["update_id"] + 1
                            await self.handle_update(update)
            except Exception as e:
                logger.error(f"Error polling updates: {e}")
                await asyncio.sleep(5)

    async def handle_update(self, update: Dict):
        """Handle incoming Telegram update"""
        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")

        if not chat_id or not text:
            return

        if text.startswith("/start"):
            await self.send_welcome_message(str(chat_id))
        elif text.startswith("/help"):
            await self.send_help_message(str(chat_id))

    async def send_welcome_message(self, chat_id: str):
        """Send welcome message with disclaimer to new users"""
        message = """
👋 *Welcome to Quantum Trading AI!*

I provide AI-powered crypto trading signals using machine learning analysis.

📊 *What I offer:*
• Signals for 16+ cryptocurrencies
• Entry, Stop Loss, and Take Profit levels
• 4-hourly market scans
• Risk/reward analysis

📢 *Join our channel:* @QuantumTradingAIX

⚠️ *IMPORTANT RISK WARNING:*
Trading signals are for *educational purposes only*. This is *not financial advice*. You can lose money trading cryptocurrencies. Past performance does not guarantee future results.

_By using this service, you acknowledge that:_
• You accept all trading risks
• You will not trade money you cannot afford to lose
• You understand this is not financial advice
• The service is provided "as-is" with no guarantees

*Trade at your own risk.*

Type /help for available commands.
"""
        await self.send_message(chat_id, message.strip())

    async def send_help_message(self, chat_id: str):
        """Send help message"""
        message = """
*Quantum Trading AI Commands*

/start - Welcome message and disclaimer
/help - Show this help message

📢 *Channel:* @QuantumTradingAIX
🤖 *Bot:* @QuantumAIXRobot

Signals are automatically posted to the channel when trading opportunities are detected.

⚠️ _Not financial advice. Trade responsibly._
"""
        await self.send_message(chat_id, message.strip())
