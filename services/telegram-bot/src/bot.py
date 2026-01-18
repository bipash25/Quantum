"""
Quantum Trading AI - Telegram Bot
=================================
Sends trading signals to Telegram channel and handles user commands.

Features:
- Subscribe to Redis for real-time signal updates
- Format signals as beautiful Telegram messages
- Handle user commands (/start, /signals, /preferences, /symbols, /timeframes, etc.)
- Track signal delivery for analytics
- Filter signals based on user preferences
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List

import aiohttp
import redis.asyncio as redis
from sqlalchemy import create_engine, text

from .config import settings, AVAILABLE_SYMBOLS, AVAILABLE_TIMEFRAMES


logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for signal delivery with user preferences"""

    BASE_URL = "https://api.telegram.org/bot"

    def __init__(self):
        self.token = settings.telegram_token
        self.channel_id = settings.telegram_channel_id
        self.api_url = f"{self.BASE_URL}{self.token}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.redis: Optional[redis.Redis] = None
        self.engine = create_engine(settings.database_url)

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

    # =========================================================================
    # USER PREFERENCES DATABASE OPERATIONS
    # =========================================================================

    def get_or_create_user(self, telegram_id: int, username: Optional[str] = None) -> Dict:
        """Get user from database or create if not exists."""
        with self.engine.begin() as conn:
            # Try to get existing user
            result = conn.execute(text("""
                SELECT id, telegram_id, telegram_username, preferred_symbols,
                       preferred_timeframes, tier, notifications_enabled
                FROM users WHERE telegram_id = :tid
            """), {"tid": telegram_id})
            row = result.fetchone()

            if row:
                return {
                    "id": row[0],
                    "telegram_id": row[1],
                    "username": row[2],
                    "preferred_symbols": row[3] or [],
                    "preferred_timeframes": row[4] or ["4h", "1d"],
                    "tier": row[5],
                    "notifications_enabled": row[6],
                }

            # Create new user
            conn.execute(text("""
                INSERT INTO users (telegram_id, telegram_username, preferred_symbols, preferred_timeframes, tier)
                VALUES (:tid, :username, '[]'::jsonb, '["4h", "1d"]'::jsonb, 'free')
                ON CONFLICT (telegram_id) DO NOTHING
            """), {"tid": telegram_id, "username": username})

            return {
                "id": None,
                "telegram_id": telegram_id,
                "username": username,
                "preferred_symbols": [],
                "preferred_timeframes": ["4h", "1d"],
                "tier": "free",
                "notifications_enabled": True,
            }

    def update_user_symbols(self, telegram_id: int, symbols: List[str]) -> bool:
        """Update user's preferred symbols."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE users
                    SET preferred_symbols = :symbols::jsonb
                    WHERE telegram_id = :tid
                """), {"tid": telegram_id, "symbols": json.dumps(symbols)})
            return True
        except Exception as e:
            logger.error(f"Error updating symbols: {e}")
            return False

    def update_user_timeframes(self, telegram_id: int, timeframes: List[str]) -> bool:
        """Update user's preferred timeframes."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE users
                    SET preferred_timeframes = :timeframes::jsonb
                    WHERE telegram_id = :tid
                """), {"tid": telegram_id, "timeframes": json.dumps(timeframes)})
            return True
        except Exception as e:
            logger.error(f"Error updating timeframes: {e}")
            return False

    def get_users_for_signal(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get all users who want to receive signals for given symbol/timeframe."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT telegram_id, tier
                FROM users
                WHERE notifications_enabled = true
                  AND (
                    preferred_symbols = '[]'::jsonb
                    OR preferred_symbols @> :symbol::jsonb
                  )
                  AND (
                    preferred_timeframes = '[]'::jsonb
                    OR preferred_timeframes @> :timeframe::jsonb
                  )
            """), {
                "symbol": json.dumps([symbol]),
                "timeframe": json.dumps([timeframe])
            })
            return [{"telegram_id": row[0], "tier": row[1]} for row in result.fetchall()]

    # =========================================================================
    # TELEGRAM API OPERATIONS
    # =========================================================================

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
                    return True
                else:
                    logger.error(f"Failed to send message: {data}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
        return False

    def format_signal(self, signal: Dict) -> str:
        """Format signal as Telegram message with clear timeframe label"""
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

        # Determine timeframe label and trade type
        if timeframe.lower() in ["1d", "24h"]:
            tf_label = "[24H SIGNAL]"
            trade_type = "Position Trade"
        else:
            tf_label = "[4H SIGNAL]"
            trade_type = "Swing Trade"

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
{tf_label} {emoji} *{direction}* | {symbol} {conf_indicator}

📊 *Timeframe:* {timeframe.upper()} ({trade_type})
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

    async def send_signal_to_users(self, signal: Dict):
        """Send signal to individual users based on their preferences."""
        symbol = signal.get("symbol", "")
        timeframe = signal.get("timeframe", "4h")

        # Get users who want this signal
        users = self.get_users_for_signal(symbol, timeframe)

        if not users:
            logger.debug(f"No users subscribed to {symbol} {timeframe}")
            return

        message = self.format_signal(signal)

        # Send to each user (rate limited)
        for user in users:
            try:
                await self.send_message(str(user["telegram_id"]), message)
                await asyncio.sleep(0.05)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to send to user {user['telegram_id']}: {e}")

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

                    # Send to channel (public)
                    await self.send_signal_to_channel(signal)

                    # Send to subscribed users (filtered by preferences)
                    await self.send_signal_to_users(signal)

                except Exception as e:
                    logger.error(f"Error processing signal: {e}")

    async def send_startup_message(self):
        """Send startup notification with disclaimer"""
        message = """
🚀 *Quantum Trading AI* is now online!

📊 Monitoring 16 crypto pairs for trading opportunities
⏱ Timeframes: 4H (swing trades) + 24H (position trades)
🎯 Strategy: ML-powered technical analysis

*Signal Types:*
• [1H SIGNAL] - Day trades, valid for 1 hour
• [4H SIGNAL] - Swing trades, valid for 4 hours
• [24H SIGNAL] - Position trades, valid for 24 hours

*DM Commands:*
/preferences - View your settings
/symbols BTC,ETH,SOL - Filter by symbols
/timeframes 1H,4H,24H - Filter by timeframe

Stay tuned for signals! 📈

⚠️ *RISK WARNING:* Trading signals are for educational purposes only. Not financial advice. You can lose money trading. Past performance does not guarantee future results. Trade at your own risk.

_By using this service, you accept our Terms of Service._
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
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        user = message.get("from", {})
        user_id = user.get("id")
        username = user.get("username")

        if not chat_id or not text:
            return

        # Ensure user exists in database
        if user_id:
            self.get_or_create_user(user_id, username)

        # Route commands
        if text.startswith("/start"):
            await self.send_welcome_message(str(chat_id), user_id)
        elif text.startswith("/help"):
            await self.send_help_message(str(chat_id))
        elif text.startswith("/preferences"):
            await self.send_preferences(str(chat_id), user_id)
        elif text.startswith("/symbols"):
            await self.handle_symbols_command(str(chat_id), user_id, text)
        elif text.startswith("/timeframes"):
            await self.handle_timeframes_command(str(chat_id), user_id, text)

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    async def send_welcome_message(self, chat_id: str, user_id: Optional[int] = None):
        """Send welcome message with disclaimer to new users"""
        message = """
👋 *Welcome to Quantum Trading AI!*

I provide AI-powered crypto trading signals using machine learning analysis.

📊 *What I offer:*
• Signals for 16 cryptocurrencies
• Entry, Stop Loss, and Take Profit levels
• 4H + 24H timeframe scans
• Risk/reward analysis

📢 *Join our channel:* @QuantumTradingAIX

*Customize your signals:*
/preferences - View your current settings
/symbols BTC,ETH,SOL - Only get signals for these coins
/timeframes 1H,4H,24H - Choose which timeframes to receive

⚠️ *IMPORTANT RISK WARNING:*
Trading signals are for *educational purposes only*. This is *not financial advice*. You can lose money trading cryptocurrencies. Past performance does not guarantee future results.

*Trade at your own risk.*

Type /help for all commands.
"""
        await self.send_message(chat_id, message.strip())

    async def send_help_message(self, chat_id: str):
        """Send help message"""
        message = """
*Quantum Trading AI Commands*

*Preference Commands:*
/preferences - Show your current settings
/symbols BTC,ETH,SOL - Set symbol filter (empty = all)
/timeframes 1H,4H,24H - Set timeframe filter

*Info Commands:*
/start - Welcome message and disclaimer
/help - Show this help message

*Available Symbols (50):*
BTC, ETH, SOL, BNB, XRP, ADA, DOGE, MATIC, DOT, AVAX,
LINK, UNI, ATOM, LTC, NEAR, FTM, ALGO, AAVE, SAND, MANA,
APT, ARB, OP, INJ, SUI, TIA, SEI, RUNE, RENDER, WLD,
IMX, LDO, STX, FIL, HBAR, VET, ICP, MKR, QNT, GRT,
FLOW, XLM, AXS, THETA, EGLD, APE, CHZ, EOS, CFX, ZIL

*Available Timeframes:*
1H (day trades), 4H (swing trades), 24H (position trades)

📢 *Channel:* @QuantumTradingAIX
🤖 *Bot:* @QuantumAIXRobot

⚠️ _Not financial advice. Trade responsibly._
"""
        await self.send_message(chat_id, message.strip())

    async def send_preferences(self, chat_id: str, user_id: Optional[int]):
        """Send user's current preferences"""
        if not user_id:
            await self.send_message(chat_id, "❌ Could not identify your user account.")
            return

        user = self.get_or_create_user(user_id)
        symbols = user.get("preferred_symbols", [])
        timeframes = user.get("preferred_timeframes", ["4h", "1d"])
        tier = user.get("tier", "free")
        notifications = user.get("notifications_enabled", True)

        # Format symbols display
        if not symbols:
            symbols_str = "All symbols (no filter)"
        else:
            symbols_str = ", ".join([s.replace("USDT", "") for s in symbols])

        # Format timeframes display
        tf_display = []
        if "4h" in timeframes:
            tf_display.append("4H")
        if "1d" in timeframes:
            tf_display.append("24H")
        timeframes_str = ", ".join(tf_display) if tf_display else "All timeframes"

        message = f"""
⚙️ *Your Preferences*

👤 *Tier:* {tier.upper()}
🔔 *Notifications:* {"Enabled" if notifications else "Disabled"}

📊 *Symbol Filter:*
{symbols_str}

⏱ *Timeframe Filter:*
{timeframes_str}

*Update your preferences:*
• `/symbols BTC,ETH,SOL` - Set symbols
• `/symbols clear` - Clear filter (receive all)
• `/timeframes 1H,4H,24H` - Set timeframes

_Signals matching your filters will be sent to you directly._
"""
        await self.send_message(chat_id, message.strip())

    async def handle_symbols_command(self, chat_id: str, user_id: Optional[int], text: str):
        """Handle /symbols command"""
        if not user_id:
            await self.send_message(chat_id, "❌ Could not identify your user account.")
            return

        # Parse arguments
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            # Show current symbols and usage
            user = self.get_or_create_user(user_id)
            symbols = user.get("preferred_symbols", [])
            if symbols:
                current = ", ".join([s.replace("USDT", "") for s in symbols])
            else:
                current = "All (no filter)"

            message = f"""
📊 *Symbol Preferences*

*Current:* {current}

*Usage:*
`/symbols BTC,ETH,SOL` - Only receive these
`/symbols clear` - Clear filter (receive all)

*Available (50 coins):*
Use /help to see full list
"""
            await self.send_message(chat_id, message.strip())
            return

        args = parts[1].strip().upper()

        # Handle clear command
        if args in ["CLEAR", "ALL", "RESET", "NONE"]:
            if self.update_user_symbols(user_id, []):
                await self.send_message(chat_id, "✅ Symbol filter cleared! You'll receive signals for *all symbols*.")
            else:
                await self.send_message(chat_id, "❌ Failed to update preferences. Please try again.")
            return

        # Parse symbol list
        symbols_input = [s.strip() for s in args.replace(" ", ",").split(",") if s.strip()]

        # Validate and normalize symbols
        valid_symbols = []
        invalid_symbols = []

        for sym in symbols_input:
            # Add USDT suffix if not present
            if not sym.endswith("USDT"):
                sym = sym + "USDT"

            if sym in AVAILABLE_SYMBOLS:
                valid_symbols.append(sym)
            else:
                invalid_symbols.append(sym.replace("USDT", ""))

        if not valid_symbols:
            available = ", ".join([s.replace("USDT", "") for s in AVAILABLE_SYMBOLS])
            await self.send_message(
                chat_id,
                f"❌ No valid symbols provided.\n\n*Available:* {available}"
            )
            return

        # Update preferences
        if self.update_user_symbols(user_id, valid_symbols):
            valid_display = ", ".join([s.replace("USDT", "") for s in valid_symbols])
            message = f"✅ Symbol filter updated!\n\n*Receiving signals for:* {valid_display}"

            if invalid_symbols:
                message += f"\n\n⚠️ *Ignored (not available):* {', '.join(invalid_symbols)}"

            await self.send_message(chat_id, message)
        else:
            await self.send_message(chat_id, "❌ Failed to update preferences. Please try again.")

    async def handle_timeframes_command(self, chat_id: str, user_id: Optional[int], text: str):
        """Handle /timeframes command"""
        if not user_id:
            await self.send_message(chat_id, "❌ Could not identify your user account.")
            return

        # Parse arguments
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            # Show current timeframes and usage
            user = self.get_or_create_user(user_id)
            timeframes = user.get("preferred_timeframes", ["1h", "4h", "1d"])
            tf_display = []
            if "1h" in timeframes:
                tf_display.append("1H")
            if "4h" in timeframes:
                tf_display.append("4H")
            if "1d" in timeframes:
                tf_display.append("24H")
            current = ", ".join(tf_display) if tf_display else "All"

            message = f"""
⏱ *Timeframe Preferences*

*Current:* {current}

*Usage:*
`/timeframes 1H` - Only 1-hour signals (day trades)
`/timeframes 4H` - Only 4-hour signals (swing trades)
`/timeframes 24H` - Only 24-hour signals (position trades)
`/timeframes 1H,4H,24H` - All timeframes
`/timeframes all` - All timeframes

*Available:*
• 1H - Day trades (every hour)
• 4H - Swing trades (every 4 hours)
• 24H - Position trades (daily)
"""
            await self.send_message(chat_id, message.strip())
            return

        args = parts[1].strip().upper()

        # Handle all command
        if args in ["ALL", "BOTH", "CLEAR"]:
            if self.update_user_timeframes(user_id, ["1h", "4h", "1d"]):
                await self.send_message(chat_id, "✅ Timeframe filter set to *all timeframes* (1H + 4H + 24H).")
            else:
                await self.send_message(chat_id, "❌ Failed to update preferences. Please try again.")
            return

        # Parse timeframe list
        tf_input = [t.strip() for t in args.replace(" ", ",").split(",") if t.strip()]

        # Normalize timeframes
        valid_timeframes = []
        for tf in tf_input:
            tf_lower = tf.lower()
            if tf_lower in ["1h", "1"]:
                if "1h" not in valid_timeframes:
                    valid_timeframes.append("1h")
            elif tf_lower in ["4h", "4"]:
                if "4h" not in valid_timeframes:
                    valid_timeframes.append("4h")
            elif tf_lower in ["24h", "1d", "24", "d", "daily"]:
                if "1d" not in valid_timeframes:
                    valid_timeframes.append("1d")

        if not valid_timeframes:
            await self.send_message(
                chat_id,
                "❌ No valid timeframes provided.\n\n*Available:* 1H, 4H, 24H"
            )
            return

        # Update preferences
        if self.update_user_timeframes(user_id, valid_timeframes):
            tf_display = []
            if "1h" in valid_timeframes:
                tf_display.append("1H (day trades)")
            if "4h" in valid_timeframes:
                tf_display.append("4H (swing trades)")
            if "1d" in valid_timeframes:
                tf_display.append("24H (position trades)")

            message = f"✅ Timeframe filter updated!\n\n*Receiving:* {', '.join(tf_display)}"
            await self.send_message(chat_id, message)
        else:
            await self.send_message(chat_id, "❌ Failed to update preferences. Please try again.")
