#!/usr/bin/env python3
"""
Quantum Trading AI - RL Training Data Export
=============================================
Exports complete state-action-reward sequences for reinforcement learning training.

Outputs:
- JSON file with RL-ready training data
- Each record contains: state, action, reward, next_state, done, info

Usage:
    python scripts/export_rl_data.py --output data/rl_training.json
    python scripts/export_rl_data.py --format jsonl --output data/rl_training.jsonl
    python scripts/export_rl_data.py --since 2024-01-01 --output data/rl_2024.json
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import pandas as pd
from sqlalchemy import create_engine, text

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://quantum:quantum_secure_password_2024@localhost:5432/quantum_trading"  # pragma: allowlist secret
)


class RLDataExporter:
    """Export RL training data from database."""

    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url)

    def get_closed_signals(self, since: Optional[str] = None) -> pd.DataFrame:
        """Get all closed signals with their outcomes."""
        query = """
            SELECT
                s.id as signal_id,
                s.created_at as signal_created_at,
                s.symbol,
                s.direction,
                s.timeframe,
                s.confidence,
                s.entry_price,
                s.stop_loss,
                s.take_profit_1,
                s.take_profit_2,
                s.take_profit_3,
                s.risk_reward_ratio,
                s.status,
                o.outcome,
                o.exit_price,
                o.actual_pnl_percent,
                o.max_favorable_excursion,
                o.max_adverse_excursion,
                o.duration_minutes,
                o.tp1_hit,
                o.tp2_hit,
                o.tp3_hit
            FROM signals s
            JOIN signal_outcomes o ON s.id = o.signal_id AND s.created_at = o.signal_created_at
            WHERE o.outcome IN ('win', 'loss', 'partial', 'expired')
        """
        params = {}
        if since:
            query += " AND s.created_at >= :since"
            params["since"] = since

        query += " ORDER BY s.created_at ASC"

        return pd.read_sql(text(query), self.engine, params=params)

    def get_signal_features(self, signal_id: int, signal_created_at: datetime) -> Dict[str, float]:
        """Get feature values for a signal."""
        query = text("""
            SELECT feature_name, feature_value
            FROM signal_features
            WHERE signal_id = :signal_id AND signal_created_at = :created_at
        """)
        result = pd.read_sql(query, self.engine, params={
            "signal_id": signal_id,
            "created_at": signal_created_at
        })
        return dict(zip(result["feature_name"], result["feature_value"]))

    def get_trade_executions(self, signal_id: int, signal_created_at: datetime) -> List[Dict]:
        """Get all trade executions for a signal."""
        query = text("""
            SELECT action, price, quantity_pct, pnl_percent, cumulative_pnl, timestamp
            FROM trade_executions
            WHERE signal_id = :signal_id AND signal_created_at = :created_at
            ORDER BY timestamp ASC
        """)
        result = pd.read_sql(query, self.engine, params={
            "signal_id": signal_id,
            "created_at": signal_created_at
        })
        return result.to_dict("records")

    def get_trade_context(self, signal_id: int, signal_created_at: datetime, context_type: str) -> Optional[Dict]:
        """Get entry or exit context for a signal."""
        query = text("""
            SELECT volatility_regime, trend_direction, volume_profile,
                   atr_value, atr_percent, volume_ratio, rsi_value, price
            FROM trade_context
            WHERE signal_id = :signal_id
              AND signal_created_at = :created_at
              AND context_type = :context_type
            LIMIT 1
        """)
        result = pd.read_sql(query, self.engine, params={
            "signal_id": signal_id,
            "created_at": signal_created_at,
            "context_type": context_type
        })
        if result.empty:
            return None
        return result.iloc[0].to_dict()

    def calculate_reward(self, outcome: str, actual_pnl: float, direction: str) -> float:
        """
        Calculate RL reward based on outcome and P&L.

        Reward design:
        - Win (all TPs hit): actual_pnl (positive)
        - Partial win: actual_pnl (positive but smaller)
        - Loss (SL hit): actual_pnl (negative, typically -2%)
        - Expired: actual_pnl (could be positive or negative)

        We use actual P&L directly as reward since it already incorporates:
        - Position sizing (2% risk)
        - Partial fills (50/30/20 at TPs)
        """
        return actual_pnl if actual_pnl else 0.0

    def export_signal_as_rl(self, signal: pd.Series) -> Optional[Dict[str, Any]]:
        """Convert a single signal to RL training format."""
        signal_id = signal["signal_id"]
        signal_created_at = signal["signal_created_at"]

        # Get state (features at entry time)
        features = self.get_signal_features(signal_id, signal_created_at)
        if not features:
            logger.warning(f"No features found for signal {signal_id}")
            return None

        # Get contexts
        entry_context = self.get_trade_context(signal_id, signal_created_at, "entry")
        exit_context = self.get_trade_context(signal_id, signal_created_at, "exit")

        # Get trade executions
        executions = self.get_trade_executions(signal_id, signal_created_at)

        # Calculate reward
        reward = self.calculate_reward(
            signal["outcome"],
            signal["actual_pnl_percent"],
            signal["direction"]
        )

        # Build RL record
        rl_record = {
            "signal_id": signal_id,
            "timestamp": signal_created_at.isoformat() if hasattr(signal_created_at, 'isoformat') else str(signal_created_at),
            "symbol": signal["symbol"],
            "timeframe": signal["timeframe"],

            # State: feature vector at entry time
            "state": features,

            # Action: trading decision
            "action": signal["direction"],  # LONG or SHORT

            # Reward: actual P&L achieved
            "reward": reward,

            # Next state: exit context (simplified)
            "next_state": exit_context if exit_context else {},

            # Episode complete
            "done": True,

            # Additional info for debugging/analysis
            "info": {
                "confidence": float(signal["confidence"]) if signal["confidence"] else None,
                "entry_price": float(signal["entry_price"]) if signal["entry_price"] else None,
                "exit_price": float(signal["exit_price"]) if signal["exit_price"] else None,
                "stop_loss": float(signal["stop_loss"]) if signal["stop_loss"] else None,
                "take_profit_1": float(signal["take_profit_1"]) if signal["take_profit_1"] else None,
                "take_profit_2": float(signal["take_profit_2"]) if signal["take_profit_2"] else None,
                "take_profit_3": float(signal["take_profit_3"]) if signal["take_profit_3"] else None,
                "risk_reward_ratio": float(signal["risk_reward_ratio"]) if signal["risk_reward_ratio"] else None,
                "outcome": signal["outcome"],
                "max_favorable_excursion": float(signal["max_favorable_excursion"]) if signal["max_favorable_excursion"] else None,
                "max_adverse_excursion": float(signal["max_adverse_excursion"]) if signal["max_adverse_excursion"] else None,
                "duration_minutes": int(signal["duration_minutes"]) if signal["duration_minutes"] else None,
                "tp1_hit": bool(signal["tp1_hit"]) if signal["tp1_hit"] is not None else False,
                "tp2_hit": bool(signal["tp2_hit"]) if signal["tp2_hit"] is not None else False,
                "tp3_hit": bool(signal["tp3_hit"]) if signal["tp3_hit"] is not None else False,
                "entry_context": entry_context,
                "exit_context": exit_context,
                "executions": [
                    {
                        "action": e["action"],
                        "price": float(e["price"]) if e["price"] else None,
                        "quantity_pct": float(e["quantity_pct"]) if e["quantity_pct"] else None,
                        "pnl_percent": float(e["pnl_percent"]) if e["pnl_percent"] else None,
                        "cumulative_pnl": float(e["cumulative_pnl"]) if e["cumulative_pnl"] else None,
                        "timestamp": e["timestamp"].isoformat() if hasattr(e["timestamp"], 'isoformat') else str(e["timestamp"]) if e["timestamp"] else None,
                    }
                    for e in executions
                ] if executions else [],
            }
        }

        return rl_record

    def export_all(
        self,
        output_path: str,
        since: Optional[str] = None,
        format: str = "json"
    ) -> int:
        """Export all closed signals as RL training data."""
        logger.info(f"Fetching closed signals{' since ' + since if since else ''}...")
        signals = self.get_closed_signals(since)
        logger.info(f"Found {len(signals)} closed signals")

        rl_records = []
        for idx, signal in signals.iterrows():
            try:
                record = self.export_signal_as_rl(signal)
                if record:
                    rl_records.append(record)
            except Exception as e:
                logger.error(f"Error exporting signal {signal['signal_id']}: {e}")

        logger.info(f"Exported {len(rl_records)} RL training records")

        # Save to file
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for record in rl_records:
                    f.write(json.dumps(record, default=str) + "\n")
        else:
            with open(output_path, "w") as f:
                json.dump(rl_records, f, indent=2, default=str)

        logger.info(f"Saved RL training data to {output_path}")
        return len(rl_records)

    def print_summary(self, since: Optional[str] = None):
        """Print summary statistics of available RL data."""
        signals = self.get_closed_signals(since)

        if signals.empty:
            print("No closed signals found.")
            return

        print("\n" + "=" * 60)
        print("RL TRAINING DATA SUMMARY")
        print("=" * 60)

        print(f"\nTotal closed signals: {len(signals)}")
        print(f"\nOutcome distribution:")
        for outcome, count in signals["outcome"].value_counts().items():
            pct = count / len(signals) * 100
            print(f"  {outcome:10s}: {count:5d} ({pct:5.1f}%)")

        print(f"\nDirection distribution:")
        for direction, count in signals["direction"].value_counts().items():
            pct = count / len(signals) * 100
            print(f"  {direction:10s}: {count:5d} ({pct:5.1f}%)")

        print(f"\nSymbol distribution (top 10):")
        for symbol, count in signals["symbol"].value_counts().head(10).items():
            pct = count / len(signals) * 100
            print(f"  {symbol:12s}: {count:5d} ({pct:5.1f}%)")

        print(f"\nTimeframe distribution:")
        for tf, count in signals["timeframe"].value_counts().items():
            pct = count / len(signals) * 100
            print(f"  {tf:10s}: {count:5d} ({pct:5.1f}%)")

        # Reward statistics
        rewards = signals["actual_pnl_percent"].dropna()
        if not rewards.empty:
            print(f"\nReward (P&L) statistics:")
            print(f"  Mean:   {rewards.mean():+.2f}%")
            print(f"  Std:    {rewards.std():.2f}%")
            print(f"  Min:    {rewards.min():+.2f}%")
            print(f"  Max:    {rewards.max():+.2f}%")
            print(f"  Total:  {rewards.sum():+.2f}%")

        # TP hit rates
        print(f"\nTP hit rates:")
        print(f"  TP1: {signals['tp1_hit'].sum():5d} / {len(signals)} ({signals['tp1_hit'].mean() * 100:.1f}%)")
        print(f"  TP2: {signals['tp2_hit'].sum():5d} / {len(signals)} ({signals['tp2_hit'].mean() * 100:.1f}%)")
        print(f"  TP3: {signals['tp3_hit'].sum():5d} / {len(signals)} ({signals['tp3_hit'].mean() * 100:.1f}%)")

        # Check feature coverage
        query = text("""
            SELECT COUNT(DISTINCT signal_id) as signals_with_features
            FROM signal_features
        """)
        with self.engine.connect() as conn:
            features_count = conn.execute(query).scalar() or 0

        print(f"\nSignals with features: {features_count} / {len(signals)} ({features_count / len(signals) * 100:.1f}%)")

        # Check execution coverage
        query = text("""
            SELECT COUNT(DISTINCT signal_id) as signals_with_executions
            FROM trade_executions
        """)
        with self.engine.connect() as conn:
            executions_count = conn.execute(query).scalar() or 0

        print(f"Signals with executions: {executions_count} / {len(signals)} ({executions_count / len(signals) * 100:.1f}%)")

        # Check context coverage
        query = text("""
            SELECT context_type, COUNT(DISTINCT signal_id) as count
            FROM trade_context
            GROUP BY context_type
        """)
        result = pd.read_sql(query, self.engine)
        if not result.empty:
            print(f"\nContext coverage:")
            for _, row in result.iterrows():
                print(f"  {row['context_type']:10s}: {row['count']:5d}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Export RL training data from Quantum Trading AI"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/rl_training.json",
        help="Output file path (default: data/rl_training.json)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "jsonl"],
        default="json",
        help="Output format: json (array) or jsonl (line-delimited)"
    )
    parser.add_argument(
        "--since",
        help="Only export signals since this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Print summary statistics without exporting"
    )
    parser.add_argument(
        "--database-url",
        default=DATABASE_URL,
        help="Database connection URL"
    )

    args = parser.parse_args()

    exporter = RLDataExporter(args.database_url)

    if args.summary:
        exporter.print_summary(args.since)
    else:
        count = exporter.export_all(
            output_path=args.output,
            since=args.since,
            format=args.format
        )
        print(f"\nExported {count} RL training records to {args.output}")


if __name__ == "__main__":
    main()
