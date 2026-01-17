#!/usr/bin/env python3
"""
Quantum Trading AI - Feature Analysis Script
=============================================
Analyzes feature importance and predictive power for RL foundation.

Usage:
    python scripts/analyze_features.py                    # Full analysis
    python scripts/analyze_features.py --export-rl        # Export RL training data
    python scripts/analyze_features.py --drift            # Check for feature drift
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "database": os.environ.get("DB_NAME", "quantum_trading"),
}


def get_engine():
    """Create database engine."""
    url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(url)


def load_feature_data(engine, days: int = 30) -> pd.DataFrame:
    """Load features with signal outcomes."""
    query = text("""
        SELECT
            sf.signal_id,
            sf.feature_name,
            sf.feature_value,
            sf.feature_importance,
            sf.feature_rank,
            s.symbol,
            s.direction,
            s.timeframe,
            s.confidence,
            so.outcome,
            so.pnl_percent,
            so.actual_pnl_percent,
            s.created_at as signal_time
        FROM signal_features sf
        JOIN signals s ON sf.signal_id = s.id AND sf.signal_created_at = s.created_at
        LEFT JOIN signal_outcomes so ON s.id = so.signal_id AND s.created_at = so.signal_created_at
        WHERE s.created_at > :since
        ORDER BY s.created_at DESC, sf.feature_name
    """)

    since = datetime.now(timezone.utc) - timedelta(days=days)
    return pd.read_sql(query, engine, params={"since": since})


def analyze_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze feature importance from model data."""
    if df.empty:
        return pd.DataFrame()

    importance = df.groupby("feature_name").agg({
        "feature_importance": "mean",
        "feature_rank": "mean",
        "signal_id": "count"
    }).reset_index()

    importance.columns = ["feature_name", "avg_importance", "avg_rank", "signal_count"]
    importance = importance.sort_values("avg_importance", ascending=False)

    return importance


def analyze_win_loss_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare feature values between winning and losing signals."""
    if df.empty or df["outcome"].isna().all():
        return pd.DataFrame()

    # Filter to completed signals only
    completed = df[df["outcome"].isin(["win", "loss", "partial"])]
    if completed.empty:
        return pd.DataFrame()

    # Mark wins (including partial as positive)
    completed = completed.copy()
    completed["is_win"] = completed["outcome"].isin(["win", "partial"])

    # Group by feature and calculate stats for wins vs losses
    results = []
    for fname in completed["feature_name"].unique():
        fdata = completed[completed["feature_name"] == fname]

        wins = fdata[fdata["is_win"]]["feature_value"]
        losses = fdata[~fdata["is_win"]]["feature_value"]

        if len(wins) < 2 or len(losses) < 2:
            continue

        # T-test for difference
        try:
            t_stat, p_value = stats.ttest_ind(wins, losses)
        except:
            t_stat, p_value = 0, 1

        results.append({
            "feature_name": fname,
            "win_mean": wins.mean(),
            "win_std": wins.std(),
            "loss_mean": losses.mean(),
            "loss_std": losses.std(),
            "difference": wins.mean() - losses.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "win_count": len(wins),
            "loss_count": len(losses),
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("p_value")

    return result_df


def analyze_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation between features and signal success."""
    if df.empty or df["outcome"].isna().all():
        return pd.DataFrame()

    # Create binary success indicator
    completed = df[df["outcome"].isin(["win", "loss", "partial"])].copy()
    completed["success"] = completed["outcome"].isin(["win", "partial"]).astype(int)

    # Pivot to get features as columns
    pivot = completed.pivot_table(
        index="signal_id",
        columns="feature_name",
        values="feature_value",
        aggfunc="first"
    )

    # Get success for each signal
    success_map = completed.groupby("signal_id")["success"].first()

    results = []
    for col in pivot.columns:
        valid = pivot[col].dropna()
        if len(valid) < 5:
            continue

        success_vals = success_map.loc[valid.index]
        try:
            corr, p_value = stats.pearsonr(valid, success_vals)
        except:
            corr, p_value = 0, 1

        results.append({
            "feature_name": col,
            "correlation": corr,
            "p_value": p_value,
            "abs_correlation": abs(corr),
            "significant": p_value < 0.05,
            "sample_size": len(valid),
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("abs_correlation", ascending=False)

    return result_df


def detect_feature_drift(df: pd.DataFrame, recent_days: int = 7) -> pd.DataFrame:
    """Detect feature drift between recent and previous period."""
    if df.empty:
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=recent_days)
    previous_cutoff = now - timedelta(days=recent_days * 2)

    # Split into recent and previous periods
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    recent = df[df["signal_time"] >= recent_cutoff]
    previous = df[(df["signal_time"] >= previous_cutoff) & (df["signal_time"] < recent_cutoff)]

    if recent.empty or previous.empty:
        logger.warning("Not enough data for drift detection")
        return pd.DataFrame()

    results = []
    for fname in df["feature_name"].unique():
        recent_vals = recent[recent["feature_name"] == fname]["feature_value"]
        previous_vals = previous[previous["feature_name"] == fname]["feature_value"]

        if len(recent_vals) < 3 or len(previous_vals) < 3:
            continue

        # Kolmogorov-Smirnov test for distribution difference
        try:
            ks_stat, p_value = stats.ks_2samp(recent_vals, previous_vals)
        except:
            ks_stat, p_value = 0, 1

        # Calculate drift magnitude
        mean_change = (recent_vals.mean() - previous_vals.mean()) / (previous_vals.std() + 1e-10)

        results.append({
            "feature_name": fname,
            "recent_mean": recent_vals.mean(),
            "previous_mean": previous_vals.mean(),
            "mean_change_std": mean_change,
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_detected": p_value < 0.05,
            "drift_severity": "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low",
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("p_value")

    return result_df


def export_rl_data(engine, output_path: str = "data/rl_training.json"):
    """Export data in RL-ready format: (state, action, reward) tuples."""
    query = text("""
        SELECT
            s.id as signal_id,
            s.symbol,
            s.direction,
            s.timeframe,
            s.confidence,
            so.outcome,
            so.actual_pnl_percent as reward
        FROM signals s
        JOIN signal_outcomes so ON s.id = so.signal_id AND s.created_at = so.signal_created_at
        WHERE so.outcome IN ('win', 'loss', 'partial', 'expired')
        ORDER BY s.created_at
    """)

    signals_df = pd.read_sql(query, engine)
    if signals_df.empty:
        logger.warning("No completed signals for RL export")
        return

    # Get features for each signal
    feature_query = text("""
        SELECT signal_id, feature_name, feature_value
        FROM signal_features
        WHERE signal_id = ANY(:ids)
    """)

    features_df = pd.read_sql(feature_query, engine, params={"ids": signals_df["signal_id"].tolist()})

    # Build RL tuples
    rl_data = []
    for _, signal in signals_df.iterrows():
        signal_features = features_df[features_df["signal_id"] == signal["signal_id"]]

        # State: feature vector
        state = {}
        for _, f in signal_features.iterrows():
            state[f["feature_name"]] = f["feature_value"]

        # Action: direction (0=LONG, 1=SHORT)
        action = 0 if signal["direction"] == "LONG" else 1

        # Reward: actual P&L or derived from outcome
        reward = signal["reward"]
        if pd.isna(reward):
            if signal["outcome"] == "win":
                reward = 2.0  # Default win reward
            elif signal["outcome"] == "partial":
                reward = 1.0
            elif signal["outcome"] == "loss":
                reward = -2.0
            else:
                reward = 0.0

        rl_data.append({
            "signal_id": int(signal["signal_id"]),
            "symbol": signal["symbol"],
            "timeframe": signal["timeframe"],
            "state": state,
            "action": action,
            "action_name": signal["direction"],
            "reward": float(reward),
            "outcome": signal["outcome"],
        })

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rl_data, f, indent=2)

    logger.info(f"Exported {len(rl_data)} RL training samples to {output_path}")
    return rl_data


def print_analysis_report(
    importance_df: pd.DataFrame,
    winloss_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    drift_df: pd.DataFrame
):
    """Print comprehensive analysis report."""
    print("\n" + "=" * 60)
    print("QUANTUM TRADING AI - FEATURE ANALYSIS REPORT")
    print("=" * 60)

    # Feature Importance
    print("\n### TOP 10 MOST IMPORTANT FEATURES (by model importance)")
    if not importance_df.empty:
        top10 = importance_df.head(10)
        for _, row in top10.iterrows():
            print(f"  {row['feature_name']:30s} importance={row['avg_importance']:.4f} rank={row['avg_rank']:.1f}")
    else:
        print("  No feature importance data available")

    # Bottom 10 (candidates for removal)
    print("\n### BOTTOM 10 LEAST IMPORTANT FEATURES (candidates for removal)")
    if not importance_df.empty and len(importance_df) > 10:
        bottom10 = importance_df.tail(10)
        for _, row in bottom10.iterrows():
            print(f"  {row['feature_name']:30s} importance={row['avg_importance']:.4f}")
    else:
        print("  Not enough data")

    # Win/Loss Comparison
    print("\n### SIGNIFICANT FEATURE DIFFERENCES (win vs loss)")
    if not winloss_df.empty:
        significant = winloss_df[winloss_df["significant"]].head(10)
        for _, row in significant.iterrows():
            direction = "+" if row["difference"] > 0 else ""
            print(f"  {row['feature_name']:30s} diff={direction}{row['difference']:.4f} p={row['p_value']:.4f}")
    else:
        print("  No significant differences found (or not enough completed signals)")

    # Correlations
    print("\n### TOP 10 FEATURES CORRELATED WITH SUCCESS")
    if not correlation_df.empty:
        top_corr = correlation_df.head(10)
        for _, row in top_corr.iterrows():
            direction = "+" if row["correlation"] > 0 else ""
            sig = "*" if row["significant"] else ""
            print(f"  {row['feature_name']:30s} r={direction}{row['correlation']:.3f}{sig}")
    else:
        print("  No correlation data available")

    # Feature Drift
    print("\n### FEATURE DRIFT ALERTS")
    if not drift_df.empty:
        drifting = drift_df[drift_df["drift_detected"]]
        if not drifting.empty:
            for _, row in drifting.iterrows():
                print(f"  [!] {row['feature_name']:30s} severity={row['drift_severity']} p={row['p_value']:.4f}")
        else:
            print("  No significant feature drift detected")
    else:
        print("  Not enough data for drift analysis")

    print("\n" + "=" * 60)


def main(args):
    """Main analysis function."""
    engine = get_engine()

    if args.export_rl:
        export_rl_data(engine, args.output or "data/rl_training.json")
        return

    logger.info(f"Loading feature data for last {args.days} days...")
    df = load_feature_data(engine, args.days)

    if df.empty:
        logger.error("No feature data found. Make sure signals have been generated with feature storage.")
        return

    logger.info(f"Loaded {len(df)} feature records for {df['signal_id'].nunique()} signals")

    # Run analyses
    logger.info("Analyzing feature importance...")
    importance_df = analyze_feature_importance(df)

    logger.info("Comparing win/loss features...")
    winloss_df = analyze_win_loss_comparison(df)

    logger.info("Calculating feature correlations...")
    correlation_df = analyze_feature_correlations(df)

    logger.info("Detecting feature drift...")
    drift_df = detect_feature_drift(df)

    # Print report
    print_analysis_report(importance_df, winloss_df, correlation_df, drift_df)

    # Export to JSON if requested
    if args.json:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_days": args.days,
            "total_signals": int(df["signal_id"].nunique()),
            "top_features": importance_df.head(10).to_dict(orient="records") if not importance_df.empty else [],
            "bottom_features": importance_df.tail(10).to_dict(orient="records") if not importance_df.empty else [],
            "significant_differences": winloss_df[winloss_df["significant"]].to_dict(orient="records") if not winloss_df.empty else [],
            "correlations": correlation_df.head(15).to_dict(orient="records") if not correlation_df.empty else [],
            "drift_alerts": drift_df[drift_df["drift_detected"]].to_dict(orient="records") if not drift_df.empty else [],
        }

        output_path = args.output or "data/feature_analysis.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze feature importance and predictive power")

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of data to analyze (default: 30)"
    )
    parser.add_argument(
        "--export-rl",
        action="store_true",
        help="Export RL training data (state, action, reward)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export analysis report as JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path"
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Focus on feature drift analysis"
    )

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("Analysis cancelled")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
