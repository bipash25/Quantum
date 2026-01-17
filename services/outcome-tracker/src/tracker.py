"""
Quantum Trading AI - Signal Outcome Tracker
============================================
Monitors active signals and tracks if TP or SL was hit.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp
import pandas as pd
from sqlalchemy import create_engine, text

from .config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class OutcomeTracker:
    """Tracks signal outcomes by checking if TP or SL was hit."""

    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Start the tracker."""
        self.session = aiohttp.ClientSession()
        logger.info("Outcome tracker started")

    async def stop(self):
        """Stop the tracker."""
        if self.session:
            await self.session.close()

    def get_active_signals(self) -> pd.DataFrame:
        """Get all active signals that need outcome tracking."""
        query = text("""
            SELECT
                s.id, s.created_at, s.symbol, s.direction,
                s.entry_price, s.stop_loss, s.take_profit_1,
                s.take_profit_2, s.take_profit_3, s.valid_until
            FROM signals s
            LEFT JOIN signal_outcomes o
                ON s.id = o.signal_id AND s.created_at = o.signal_created_at
            WHERE s.status = 'active'
              AND o.id IS NULL
              AND s.valid_until > NOW()
            ORDER BY s.created_at DESC
        """)
        return pd.read_sql(query, self.engine)

    def get_price_data_since(
        self, symbol: str, start_time: datetime
    ) -> pd.DataFrame:
        """Get 1-minute candles since the signal was created."""
        query = text("""
            SELECT time, high, low, close
            FROM candles
            WHERE symbol = :symbol
              AND timeframe = '1m'
              AND time >= :start_time
            ORDER BY time ASC
        """)
        return pd.read_sql(
            query, self.engine,
            params={"symbol": symbol, "start_time": start_time}
        )

    def check_signal_outcome(
        self, signal: pd.Series, candles: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Check if signal hit TP or SL.

        Returns outcome dict if resolved, None if still pending.
        """
        if candles.empty:
            return None

        direction = signal["direction"]
        entry = signal["entry_price"]
        sl = signal["stop_loss"]
        tp1 = signal["take_profit_1"]
        tp2 = signal["take_profit_2"]
        tp3 = signal["take_profit_3"]

        # Track extremes for MFE/MAE
        max_favorable = 0.0
        max_adverse = 0.0

        for _, candle in candles.iterrows():
            high = candle["high"]
            low = candle["low"]
            close = candle["close"]
            candle_time = candle["time"]

            if direction == "LONG":
                # Max favorable = highest price reached
                excursion = (high - entry) / entry * 100
                max_favorable = max(max_favorable, excursion)

                # Max adverse = lowest price reached
                adverse = (entry - low) / entry * 100
                max_adverse = max(max_adverse, adverse)

                # Check if SL hit first (use low)
                if low <= sl:
                    pnl = (sl - entry) / entry * 100
                    return {
                        "outcome": "loss",
                        "exit_price": sl,
                        "exit_time": candle_time,
                        "pnl_percent": pnl,
                        "max_favorable_excursion": max_favorable,
                        "max_adverse_excursion": max_adverse,
                    }

                # Check if TP1 hit (use high)
                if high >= tp1:
                    # For simplicity, assume TP1 hit = win
                    pnl = (tp1 - entry) / entry * 100
                    return {
                        "outcome": "win",
                        "exit_price": tp1,
                        "exit_time": candle_time,
                        "pnl_percent": pnl,
                        "max_favorable_excursion": max_favorable,
                        "max_adverse_excursion": max_adverse,
                    }

            else:  # SHORT
                # Max favorable = lowest price reached (profit for short)
                excursion = (entry - low) / entry * 100
                max_favorable = max(max_favorable, excursion)

                # Max adverse = highest price reached (loss for short)
                adverse = (high - entry) / entry * 100
                max_adverse = max(max_adverse, adverse)

                # Check if SL hit first (use high)
                if high >= sl:
                    pnl = (entry - sl) / entry * 100  # Negative
                    return {
                        "outcome": "loss",
                        "exit_price": sl,
                        "exit_time": candle_time,
                        "pnl_percent": pnl,
                        "max_favorable_excursion": max_favorable,
                        "max_adverse_excursion": max_adverse,
                    }

                # Check if TP1 hit (use low)
                if low <= tp1:
                    pnl = (entry - tp1) / entry * 100  # Positive
                    return {
                        "outcome": "win",
                        "exit_price": tp1,
                        "exit_time": candle_time,
                        "pnl_percent": pnl,
                        "max_favorable_excursion": max_favorable,
                        "max_adverse_excursion": max_adverse,
                    }

        # Check if signal expired
        if datetime.now(timezone.utc) > signal["valid_until"].replace(tzinfo=timezone.utc):
            last_close = candles.iloc[-1]["close"]
            last_time = candles.iloc[-1]["time"]

            if direction == "LONG":
                pnl = (last_close - entry) / entry * 100
            else:
                pnl = (entry - last_close) / entry * 100

            return {
                "outcome": "expired",
                "exit_price": last_close,
                "exit_time": last_time,
                "pnl_percent": pnl,
                "max_favorable_excursion": max_favorable,
                "max_adverse_excursion": max_adverse,
            }

        return None

    def save_outcome(
        self, signal_id: int, signal_created_at: datetime,
        outcome: Dict, duration_minutes: int
    ):
        """Save outcome to database."""
        insert = text("""
            INSERT INTO signal_outcomes (
                signal_id, signal_created_at, outcome, exit_price, exit_time,
                pnl_percent, max_favorable_excursion, max_adverse_excursion,
                duration_minutes
            ) VALUES (
                :signal_id, :signal_created_at, :outcome, :exit_price, :exit_time,
                :pnl_percent, :mfe, :mae, :duration
            )
        """)

        # Also update signal status
        update = text("""
            UPDATE signals
            SET status = :status
            WHERE id = :signal_id AND created_at = :signal_created_at
        """)

        status = "hit_tp" if outcome["outcome"] == "win" else "hit_sl"
        if outcome["outcome"] == "expired":
            status = "expired"

        with self.engine.begin() as conn:
            conn.execute(insert, {
                "signal_id": signal_id,
                "signal_created_at": signal_created_at,
                "outcome": outcome["outcome"],
                "exit_price": outcome["exit_price"],
                "exit_time": outcome["exit_time"],
                "pnl_percent": outcome["pnl_percent"],
                "mfe": outcome["max_favorable_excursion"],
                "mae": outcome["max_adverse_excursion"],
                "duration": duration_minutes,
            })
            conn.execute(update, {
                "signal_id": signal_id,
                "signal_created_at": signal_created_at,
                "status": status,
            })

    async def send_outcome_notification(self, signal: pd.Series, outcome: Dict):
        """Send Telegram notification about signal outcome."""
        if not settings.telegram_token:
            return

        direction = signal["direction"]
        symbol = signal["symbol"]
        entry = signal["entry_price"]
        exit_price = outcome["exit_price"]
        pnl = outcome["pnl_percent"]
        result = outcome["outcome"]

        emoji = "✅" if result == "win" else "❌" if result == "loss" else "⏰"

        message = f"""
{emoji} *Signal Outcome Update*

📊 *{symbol}* {direction}
💰 Entry: `${entry:.4f}`
🏁 Exit: `${exit_price:.4f}`
📈 P&L: `{pnl:+.2f}%`
🎯 Result: *{result.upper()}*

_📈 @QuantumTradingAIX_
"""

        url = f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage"
        try:
            async with self.session.post(url, json={
                "chat_id": settings.telegram_channel_id,
                "text": message.strip(),
                "parse_mode": "Markdown",
            }) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.error(f"Telegram error: {data}")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    async def check_all_signals(self):
        """Check all active signals for outcomes."""
        signals = self.get_active_signals()
        logger.info(f"Checking {len(signals)} active signals for outcomes")

        for _, signal in signals.iterrows():
            try:
                candles = self.get_price_data_since(
                    signal["symbol"],
                    signal["created_at"]
                )

                outcome = self.check_signal_outcome(signal, candles)

                if outcome:
                    # Calculate duration
                    created = signal["created_at"]
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    exit_time = outcome["exit_time"]
                    if hasattr(exit_time, 'tzinfo') and exit_time.tzinfo is None:
                        exit_time = exit_time.replace(tzinfo=timezone.utc)
                    elif not hasattr(exit_time, 'tzinfo'):
                        exit_time = datetime.now(timezone.utc)

                    duration = int((exit_time - created).total_seconds() / 60)

                    # Save to database
                    self.save_outcome(
                        signal["id"],
                        signal["created_at"],
                        outcome,
                        duration
                    )

                    # Send notification
                    await self.send_outcome_notification(signal, outcome)

                    logger.info(
                        f"{signal['symbol']} {signal['direction']}: "
                        f"{outcome['outcome']} ({outcome['pnl_percent']:+.2f}%)"
                    )

            except Exception as e:
                logger.error(f"Error checking {signal['symbol']}: {e}")

    async def run(self):
        """Main run loop."""
        await self.start()

        try:
            while True:
                await self.check_all_signals()
                await asyncio.sleep(settings.check_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Outcome tracker shutting down...")
        finally:
            await self.stop()


async def main():
    tracker = OutcomeTracker()
    await tracker.run()


if __name__ == "__main__":
    asyncio.run(main())
