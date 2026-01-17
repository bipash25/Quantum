"""
Quantum Trading AI - Signal Outcome Tracker (Enhanced)
=======================================================
Monitors active signals, tracks partial TP fills, calculates P&L,
and sends real-time Telegram updates.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

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


# Position sizing for P&L calculation
RISK_PER_TRADE_PCT = 2.0  # 2% risk per trade

# Partial fill percentages
TP1_POSITION_PCT = 50  # Close 50% at TP1
TP2_POSITION_PCT = 30  # Close 30% at TP2
TP3_POSITION_PCT = 20  # Close 20% at TP3


class OutcomeTracker:
    """Enhanced tracker with partial fills, P&L, and real-time updates."""

    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Start the tracker."""
        self.session = aiohttp.ClientSession()
        logger.info("Enhanced outcome tracker started")
        logger.info(f"Risk per trade: {RISK_PER_TRADE_PCT}%")
        logger.info(f"TP splits: TP1={TP1_POSITION_PCT}%, TP2={TP2_POSITION_PCT}%, TP3={TP3_POSITION_PCT}%")

    async def stop(self):
        """Stop the tracker."""
        if self.session:
            await self.session.close()

    def get_active_signals(self) -> pd.DataFrame:
        """Get all active signals (including partially filled)."""
        query = text("""
            SELECT
                s.id, s.created_at, s.symbol, s.direction, s.timeframe,
                s.entry_price, s.stop_loss, s.take_profit_1,
                s.take_profit_2, s.take_profit_3, s.valid_until,
                s.risk_reward_ratio, s.status,
                o.id as outcome_id, o.tp1_hit, o.tp2_hit, o.tp3_hit,
                o.tp1_time, o.tp2_time, o.tp3_time
            FROM signals s
            LEFT JOIN signal_outcomes o
                ON s.id = o.signal_id AND s.created_at = o.signal_created_at
            WHERE s.status IN ('active', 'hit_tp1', 'hit_tp2')
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

    def calculate_actual_pnl(
        self,
        entry: float,
        direction: str,
        tp1: float, tp2: float, tp3: float,
        sl: float,
        tp1_hit: bool, tp2_hit: bool, tp3_hit: bool,
        sl_hit: bool,
        exit_price: Optional[float] = None
    ) -> float:
        """
        Calculate actual P&L with partial fills.

        Using 2% risk per trade:
        - If SL hit on full position = -2%
        - If TP1 hit (50% at 2R) and SL hit remaining = 50%*4% - 50%*2% = +1%
        - If TP1+TP2 hit and SL remaining = 50%*4% + 30%*X% - 20%*2%
        - If all TPs hit = 50%*TP1_gain + 30%*TP2_gain + 20%*TP3_gain

        Returns actual portfolio P&L percentage.
        """
        risk = RISK_PER_TRADE_PCT

        # Calculate each TP gain in terms of R (risk multiples)
        if direction == "LONG":
            sl_distance = entry - sl
            tp1_gain = (tp1 - entry) / sl_distance if sl_distance else 0
            tp2_gain = (tp2 - entry) / sl_distance if sl_distance and tp2 else 0
            tp3_gain = (tp3 - entry) / sl_distance if sl_distance and tp3 else 0
        else:  # SHORT
            sl_distance = sl - entry
            tp1_gain = (entry - tp1) / sl_distance if sl_distance else 0
            tp2_gain = (entry - tp2) / sl_distance if sl_distance and tp2 else 0
            tp3_gain = (entry - tp3) / sl_distance if sl_distance and tp3 else 0

        total_pnl = 0.0

        if sl_hit and not tp1_hit:
            # Full stop loss
            return -risk

        if tp1_hit:
            # TP1 profit: 50% position * TP1 gain * risk
            total_pnl += (TP1_POSITION_PCT / 100) * tp1_gain * risk

        if tp2_hit:
            # TP2 profit: 30% position * TP2 gain * risk
            total_pnl += (TP2_POSITION_PCT / 100) * tp2_gain * risk

        if tp3_hit:
            # TP3 profit: 20% position * TP3 gain * risk
            total_pnl += (TP3_POSITION_PCT / 100) * tp3_gain * risk
        elif sl_hit and tp1_hit and not tp2_hit:
            # SL hit on remaining 50% after TP1
            total_pnl -= (50 / 100) * risk
        elif sl_hit and tp2_hit and not tp3_hit:
            # SL hit on remaining 20% after TP2
            total_pnl -= (20 / 100) * risk

        return total_pnl

    def check_signal_outcome(
        self, signal: pd.Series, candles: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Check if signal hit TP levels or SL.
        Tracks partial fills progressively.
        """
        if candles.empty:
            return None

        direction = signal["direction"]
        entry = signal["entry_price"]
        sl = signal["stop_loss"]
        tp1 = signal["take_profit_1"]
        tp2 = signal.get("take_profit_2") or tp1 * 1.5 if direction == "LONG" else tp1 * 0.5
        tp3 = signal.get("take_profit_3") or tp1 * 2.0 if direction == "LONG" else tp1 * 0.25

        # Current TP state from database
        tp1_hit = bool(signal.get("tp1_hit"))
        tp2_hit = bool(signal.get("tp2_hit"))
        tp3_hit = bool(signal.get("tp3_hit"))

        # Track new hits this check
        new_tp1_hit = False
        new_tp2_hit = False
        new_tp3_hit = False
        sl_hit = False

        tp1_time = None
        tp2_time = None
        tp3_time = None
        exit_time = None
        exit_price = None

        # Track extremes for MFE/MAE
        max_favorable = 0.0
        max_adverse = 0.0

        for _, candle in candles.iterrows():
            high = candle["high"]
            low = candle["low"]
            close = candle["close"]
            candle_time = candle["time"]

            if direction == "LONG":
                # MFE/MAE tracking
                excursion = (high - entry) / entry * 100
                max_favorable = max(max_favorable, excursion)
                adverse = (entry - low) / entry * 100
                max_adverse = max(max_adverse, adverse)

                # Check SL first (highest priority)
                if low <= sl and not tp3_hit:
                    sl_hit = True
                    exit_price = sl
                    exit_time = candle_time
                    break

                # Check TP levels progressively
                if not tp1_hit and high >= tp1:
                    tp1_hit = True
                    new_tp1_hit = True
                    tp1_time = candle_time

                if tp1_hit and not tp2_hit and tp2 and high >= tp2:
                    tp2_hit = True
                    new_tp2_hit = True
                    tp2_time = candle_time

                if tp2_hit and not tp3_hit and tp3 and high >= tp3:
                    tp3_hit = True
                    new_tp3_hit = True
                    tp3_time = candle_time
                    exit_price = tp3
                    exit_time = candle_time

            else:  # SHORT
                excursion = (entry - low) / entry * 100
                max_favorable = max(max_favorable, excursion)
                adverse = (high - entry) / entry * 100
                max_adverse = max(max_adverse, adverse)

                # Check SL first
                if high >= sl and not tp3_hit:
                    sl_hit = True
                    exit_price = sl
                    exit_time = candle_time
                    break

                # Check TP levels progressively
                if not tp1_hit and low <= tp1:
                    tp1_hit = True
                    new_tp1_hit = True
                    tp1_time = candle_time

                if tp1_hit and not tp2_hit and tp2 and low <= tp2:
                    tp2_hit = True
                    new_tp2_hit = True
                    tp2_time = candle_time

                if tp2_hit and not tp3_hit and tp3 and low <= tp3:
                    tp3_hit = True
                    new_tp3_hit = True
                    tp3_time = candle_time
                    exit_price = tp3
                    exit_time = candle_time

        # Determine outcome
        if sl_hit:
            outcome = "loss"
        elif tp3_hit:
            outcome = "win"
        elif tp1_hit or tp2_hit:
            outcome = "partial"
        else:
            # Check if expired
            if datetime.now(timezone.utc) > signal["valid_until"].replace(tzinfo=timezone.utc):
                outcome = "expired"
                exit_price = candles.iloc[-1]["close"] if not candles.empty else entry
                exit_time = candles.iloc[-1]["time"] if not candles.empty else datetime.now(timezone.utc)
            else:
                return None  # Still pending

        # Calculate actual P&L
        actual_pnl = self.calculate_actual_pnl(
            entry, direction, tp1, tp2, tp3, sl,
            tp1_hit, tp2_hit, tp3_hit, sl_hit, exit_price
        )

        # Calculate simple price P&L for reference
        if exit_price:
            if direction == "LONG":
                pnl_percent = (exit_price - entry) / entry * 100
            else:
                pnl_percent = (entry - exit_price) / entry * 100
        else:
            pnl_percent = 0

        return {
            "outcome": outcome,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "pnl_percent": pnl_percent,
            "actual_pnl_percent": actual_pnl,
            "max_favorable_excursion": max_favorable,
            "max_adverse_excursion": max_adverse,
            "tp1_hit": tp1_hit,
            "tp2_hit": tp2_hit,
            "tp3_hit": tp3_hit,
            "tp1_time": tp1_time,
            "tp2_time": tp2_time,
            "tp3_time": tp3_time,
            "new_tp1_hit": new_tp1_hit,
            "new_tp2_hit": new_tp2_hit,
            "new_tp3_hit": new_tp3_hit,
            "sl_hit": sl_hit,
        }

    def save_outcome(
        self, signal_id: int, signal_created_at: datetime,
        outcome: Dict, duration_minutes: int, outcome_id: Optional[int] = None
    ):
        """Save or update outcome in database."""
        if outcome_id:
            # Update existing outcome record
            update_outcome = text("""
                UPDATE signal_outcomes SET
                    outcome = :outcome,
                    exit_price = COALESCE(:exit_price, exit_price),
                    exit_time = COALESCE(:exit_time, exit_time),
                    pnl_percent = COALESCE(:pnl_percent, pnl_percent),
                    actual_pnl_percent = :actual_pnl,
                    max_favorable_excursion = :mfe,
                    max_adverse_excursion = :mae,
                    duration_minutes = :duration,
                    tp1_hit = :tp1_hit,
                    tp1_time = COALESCE(:tp1_time, tp1_time),
                    tp2_hit = :tp2_hit,
                    tp2_time = COALESCE(:tp2_time, tp2_time),
                    tp3_hit = :tp3_hit,
                    tp3_time = COALESCE(:tp3_time, tp3_time)
                WHERE id = :outcome_id
            """)
            with self.engine.begin() as conn:
                conn.execute(update_outcome, {
                    "outcome_id": outcome_id,
                    "outcome": outcome["outcome"],
                    "exit_price": outcome.get("exit_price"),
                    "exit_time": outcome.get("exit_time"),
                    "pnl_percent": outcome.get("pnl_percent"),
                    "actual_pnl": outcome.get("actual_pnl_percent"),
                    "mfe": outcome["max_favorable_excursion"],
                    "mae": outcome["max_adverse_excursion"],
                    "duration": duration_minutes,
                    "tp1_hit": outcome["tp1_hit"],
                    "tp1_time": outcome.get("tp1_time"),
                    "tp2_hit": outcome["tp2_hit"],
                    "tp2_time": outcome.get("tp2_time"),
                    "tp3_hit": outcome["tp3_hit"],
                    "tp3_time": outcome.get("tp3_time"),
                })
        else:
            # Insert new outcome record
            insert = text("""
                INSERT INTO signal_outcomes (
                    signal_id, signal_created_at, outcome, exit_price, exit_time,
                    pnl_percent, actual_pnl_percent, max_favorable_excursion,
                    max_adverse_excursion, duration_minutes, risk_amount_pct,
                    tp1_hit, tp1_time, tp2_hit, tp2_time, tp3_hit, tp3_time
                ) VALUES (
                    :signal_id, :signal_created_at, :outcome, :exit_price, :exit_time,
                    :pnl_percent, :actual_pnl, :mfe, :mae, :duration, :risk_pct,
                    :tp1_hit, :tp1_time, :tp2_hit, :tp2_time, :tp3_hit, :tp3_time
                )
            """)
            with self.engine.begin() as conn:
                conn.execute(insert, {
                    "signal_id": signal_id,
                    "signal_created_at": signal_created_at,
                    "outcome": outcome["outcome"],
                    "exit_price": outcome.get("exit_price"),
                    "exit_time": outcome.get("exit_time"),
                    "pnl_percent": outcome.get("pnl_percent"),
                    "actual_pnl": outcome.get("actual_pnl_percent"),
                    "mfe": outcome["max_favorable_excursion"],
                    "mae": outcome["max_adverse_excursion"],
                    "duration": duration_minutes,
                    "risk_pct": RISK_PER_TRADE_PCT,
                    "tp1_hit": outcome["tp1_hit"],
                    "tp1_time": outcome.get("tp1_time"),
                    "tp2_hit": outcome["tp2_hit"],
                    "tp2_time": outcome.get("tp2_time"),
                    "tp3_hit": outcome["tp3_hit"],
                    "tp3_time": outcome.get("tp3_time"),
                })

        # Update signal status
        if outcome["outcome"] == "win":
            status = "hit_tp3" if outcome["tp3_hit"] else "hit_tp"
        elif outcome["outcome"] == "partial":
            status = "hit_tp2" if outcome["tp2_hit"] else "hit_tp1"
        elif outcome["outcome"] == "loss":
            status = "hit_sl"
        else:
            status = "expired"

        update_signal = text("""
            UPDATE signals
            SET status = :status
            WHERE id = :signal_id AND created_at = :signal_created_at
        """)
        with self.engine.begin() as conn:
            conn.execute(update_signal, {
                "signal_id": signal_id,
                "signal_created_at": signal_created_at,
                "status": status,
            })

    def get_cumulative_stats(self) -> Dict:
        """Get cumulative performance statistics."""
        query = text("""
            SELECT
                COUNT(*) as total_signals,
                COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
                COUNT(CASE WHEN outcome = 'partial' THEN 1 END) as partial,
                COALESCE(SUM(actual_pnl_percent), 0) as total_pnl,
                COALESCE(AVG(actual_pnl_percent), 0) as avg_pnl
            FROM signal_outcomes
            WHERE outcome IN ('win', 'loss', 'partial', 'expired')
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchone()

        total = result[0] or 0
        wins = result[1] or 0
        losses = result[2] or 0

        return {
            "total_signals": total,
            "wins": wins,
            "losses": losses,
            "partial": result[3] or 0,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "total_pnl": result[4] or 0,
            "avg_pnl": result[5] or 0,
        }

    def get_weekly_stats(self) -> Dict:
        """Get statistics for the current week."""
        query = text("""
            SELECT
                COUNT(*) as total_signals,
                COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
                COALESCE(SUM(actual_pnl_percent), 0) as total_pnl,
                COUNT(CASE WHEN tp1_hit THEN 1 END) as tp1_hits,
                COUNT(CASE WHEN tp2_hit THEN 1 END) as tp2_hits,
                COUNT(CASE WHEN tp3_hit THEN 1 END) as tp3_hits
            FROM signal_outcomes
            WHERE created_at >= date_trunc('week', NOW())
              AND outcome IN ('win', 'loss', 'partial', 'expired')
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query).fetchone()

        total = result[0] or 0
        wins = result[1] or 0

        return {
            "total_signals": total,
            "wins": wins,
            "losses": result[2] or 0,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "total_pnl": result[3] or 0,
            "tp1_hits": result[4] or 0,
            "tp2_hits": result[5] or 0,
            "tp3_hits": result[6] or 0,
        }

    async def send_tp_notification(
        self, signal: pd.Series, tp_level: int, pnl_pct: float
    ):
        """Send Telegram notification when TP level is hit."""
        if not settings.telegram_token:
            return

        symbol = signal["symbol"].replace("USDT", "")
        direction = signal["direction"]
        timeframe = signal.get("timeframe", "4h")

        # Get cumulative stats
        stats = self.get_cumulative_stats()

        message = f"""🎯 *SIGNAL UPDATE*

*{symbol}* {direction} hit TP{tp_level} ({pnl_pct:+.1f}%)

📊 *Overall Stats:*
Signals: {stats['total_signals']} | Wins: {stats['wins']} ({stats['win_rate']:.1f}%)
Cumulative P&L: {stats['total_pnl']:+.1f}%

_[{timeframe.upper()}] @QuantumTradingAIX_"""

        await self._send_telegram(message)

    async def send_sl_notification(self, signal: pd.Series, pnl_pct: float):
        """Send Telegram notification when SL is hit."""
        if not settings.telegram_token:
            return

        symbol = signal["symbol"].replace("USDT", "")
        direction = signal["direction"]
        timeframe = signal.get("timeframe", "4h")

        stats = self.get_cumulative_stats()

        # Check if any TPs were hit before SL
        tp1_hit = signal.get("tp1_hit", False)
        tp2_hit = signal.get("tp2_hit", False)

        if tp1_hit:
            partial_msg = f" (after TP{'2' if tp2_hit else '1'})"
        else:
            partial_msg = ""

        message = f"""❌ *SIGNAL CLOSED*

*{symbol}* {direction} hit Stop Loss{partial_msg}
P&L: {pnl_pct:+.1f}%

📊 *Overall Stats:*
Signals: {stats['total_signals']} | Win Rate: {stats['win_rate']:.1f}%
Cumulative P&L: {stats['total_pnl']:+.1f}%

_[{timeframe.upper()}] @QuantumTradingAIX_"""

        await self._send_telegram(message)

    async def send_weekly_summary(self):
        """Send weekly performance summary to Telegram."""
        if not settings.telegram_token:
            return

        weekly = self.get_weekly_stats()
        cumulative = self.get_cumulative_stats()

        # Calculate week number
        week_num = datetime.now(timezone.utc).isocalendar()[1]

        message = f"""📈 *WEEK {week_num} SUMMARY*

📊 *This Week:*
Signals: {weekly['total_signals']}
Wins: {weekly['wins']} ({weekly['win_rate']:.1f}%)
TP Hits: TP1={weekly['tp1_hits']} | TP2={weekly['tp2_hits']} | TP3={weekly['tp3_hits']}
Weekly P&L: {weekly['total_pnl']:+.1f}%

📈 *All-Time:*
Total Signals: {cumulative['total_signals']}
Win Rate: {cumulative['win_rate']:.1f}%
Cumulative P&L: {cumulative['total_pnl']:+.1f}%

_🤖 @QuantumTradingAIX_"""

        await self._send_telegram(message)
        logger.info(f"Weekly summary sent: Week {week_num}")

    async def _send_telegram(self, message: str):
        """Send message to Telegram channel."""
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
                    exit_time = outcome.get("exit_time") or datetime.now(timezone.utc)
                    if hasattr(exit_time, 'tzinfo') and exit_time.tzinfo is None:
                        exit_time = exit_time.replace(tzinfo=timezone.utc)
                    duration = int((exit_time - created).total_seconds() / 60)

                    # Save to database
                    outcome_id = signal.get("outcome_id")
                    self.save_outcome(
                        signal["id"],
                        signal["created_at"],
                        outcome,
                        duration,
                        outcome_id=outcome_id if pd.notna(outcome_id) else None
                    )

                    # Send notifications for new TP hits
                    if outcome["new_tp1_hit"]:
                        entry = signal["entry_price"]
                        tp1 = signal["take_profit_1"]
                        pnl = ((tp1 - entry) / entry * 100) if signal["direction"] == "LONG" else ((entry - tp1) / entry * 100)
                        await self.send_tp_notification(signal, 1, pnl)
                        logger.info(f"{signal['symbol']} hit TP1 ({pnl:+.2f}%)")

                    if outcome["new_tp2_hit"]:
                        entry = signal["entry_price"]
                        tp2 = signal.get("take_profit_2") or signal["take_profit_1"] * 1.5
                        pnl = ((tp2 - entry) / entry * 100) if signal["direction"] == "LONG" else ((entry - tp2) / entry * 100)
                        await self.send_tp_notification(signal, 2, pnl)
                        logger.info(f"{signal['symbol']} hit TP2 ({pnl:+.2f}%)")

                    if outcome["new_tp3_hit"]:
                        entry = signal["entry_price"]
                        tp3 = signal.get("take_profit_3") or signal["take_profit_1"] * 2.0
                        pnl = ((tp3 - entry) / entry * 100) if signal["direction"] == "LONG" else ((entry - tp3) / entry * 100)
                        await self.send_tp_notification(signal, 3, pnl)
                        logger.info(f"{signal['symbol']} hit TP3 (FULL WIN) ({pnl:+.2f}%)")

                    if outcome["sl_hit"]:
                        await self.send_sl_notification(signal, outcome["actual_pnl_percent"])
                        logger.info(f"{signal['symbol']} hit SL ({outcome['actual_pnl_percent']:+.2f}%)")

            except Exception as e:
                logger.error(f"Error checking {signal['symbol']}: {e}", exc_info=True)

    async def run_weekly_summary_scheduler(self):
        """Schedule weekly summary every Sunday at 23:00 UTC."""
        while True:
            now = datetime.now(timezone.utc)
            # Calculate next Sunday 23:00 UTC
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0 and now.hour >= 23:
                days_until_sunday = 7
            next_sunday = now.replace(
                hour=23, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_until_sunday)

            wait_seconds = (next_sunday - now).total_seconds()
            logger.info(f"Next weekly summary in {wait_seconds/3600:.1f} hours")

            await asyncio.sleep(wait_seconds)
            await self.send_weekly_summary()

    async def run(self):
        """Main run loop."""
        await self.start()

        # Start weekly summary scheduler
        weekly_task = asyncio.create_task(self.run_weekly_summary_scheduler())

        try:
            while True:
                await self.check_all_signals()
                await asyncio.sleep(settings.check_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Outcome tracker shutting down...")
            weekly_task.cancel()
        finally:
            await self.stop()


async def main():
    tracker = OutcomeTracker()
    await tracker.run()


if __name__ == "__main__":
    asyncio.run(main())
