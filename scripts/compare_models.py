#!/usr/bin/env python3
"""
Quantum Trading AI - Model Version Comparison Tool
===================================================
Compare and analyze model versions stored in the database.

Usage:
    python scripts/compare_models.py                     # Show all models
    python scripts/compare_models.py --symbol BTCUSDT    # Filter by symbol
    python scripts/compare_models.py --timeframe 4h      # Filter by timeframe
    python scripts/compare_models.py --active            # Show only active models
    python scripts/compare_models.py --promote v3 BTCUSDT 4h  # Promote a model
"""
import argparse
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from tabulate import tabulate


DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    db_user = os.environ.get("DB_USER", "")
    db_pass = os.environ.get("DB_PASSWORD", "")
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "quantum_trading")
    if db_user and db_pass:
        DB_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    else:
        raise ValueError("DATABASE_URL or DB_USER/DB_PASSWORD environment variables required")


def get_models(
    engine,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    active_only: bool = False,
) -> pd.DataFrame:
    """Query model versions from database."""
    query = """
        SELECT
            version_name,
            symbol,
            timeframe,
            combined_precision,
            long_precision,
            short_precision,
            long_auc,
            short_auc,
            atr_multiplier,
            num_features,
            total_samples,
            is_active,
            created_at,
            model_path
        FROM model_versions
        WHERE 1=1
    """
    params = {}

    if symbol:
        query += " AND symbol = :symbol"
        params["symbol"] = symbol

    if timeframe:
        query += " AND timeframe = :timeframe"
        params["timeframe"] = timeframe

    if active_only:
        query += " AND is_active = true"

    query += " ORDER BY symbol, timeframe, created_at DESC"

    return pd.read_sql(text(query), engine, params=params)


def display_models(df: pd.DataFrame) -> None:
    """Display models in a formatted table."""
    if df.empty:
        print("No models found.")
        return

    # Format for display
    display_df = df.copy()
    display_df["precision"] = display_df["combined_precision"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display_df["long_prec"] = display_df["long_precision"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display_df["short_prec"] = display_df["short_precision"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display_df["auc"] = display_df.apply(
        lambda row: f"{row['long_auc']:.2f}/{row['short_auc']:.2f}"
        if pd.notna(row['long_auc']) else "N/A",
        axis=1
    )
    display_df["active"] = display_df["is_active"].apply(lambda x: "YES" if x else "")
    display_df["date"] = display_df["created_at"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M") if pd.notna(x) else "N/A"
    )

    # Select columns for display
    cols = ["version_name", "symbol", "timeframe", "precision", "long_prec", "short_prec",
            "auc", "atr_multiplier", "total_samples", "active", "date"]
    headers = ["Version", "Symbol", "TF", "Combined", "Long", "Short",
               "AUC (L/S)", "ATR Mult", "Samples", "Active", "Created"]

    print(tabulate(display_df[cols], headers=headers, tablefmt="simple", showindex=False))


def display_comparison(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """Display side-by-side comparison of model versions for a symbol/timeframe."""
    subset = df[(df["symbol"] == symbol) & (df["timeframe"] == timeframe)]

    if subset.empty:
        print(f"No models found for {symbol}/{timeframe}")
        return

    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON: {symbol} / {timeframe}")
    print(f"{'='*60}")

    # Find best model
    best_idx = subset["combined_precision"].idxmax()
    best_version = subset.loc[best_idx, "version_name"]
    best_precision = subset.loc[best_idx, "combined_precision"]

    active_model = subset[subset["is_active"] == True]
    if not active_model.empty:
        active_version = active_model.iloc[0]["version_name"]
        active_precision = active_model.iloc[0]["combined_precision"]
    else:
        active_version = None
        active_precision = 0

    print(f"\nActive Model: {active_version or 'None'} (precision: {active_precision:.1%})")
    print(f"Best Model:   {best_version} (precision: {best_precision:.1%})")

    if best_version != active_version and active_version:
        improvement = (best_precision - active_precision) / active_precision * 100
        print(f"\nRecommendation: Promote '{best_version}' (+{improvement:.1f}% improvement)")
        print(f"  Run: python scripts/compare_models.py --promote {best_version} {symbol} {timeframe}")

    display_models(subset)


def display_summary(df: pd.DataFrame) -> None:
    """Display summary statistics."""
    if df.empty:
        return

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Group by symbol/timeframe
    summary = df.groupby(["symbol", "timeframe"]).agg({
        "version_name": "count",
        "combined_precision": ["mean", "max"],
        "is_active": "sum"
    }).reset_index()

    summary.columns = ["Symbol", "TF", "Versions", "Avg Precision", "Best Precision", "Active"]
    summary["Avg Precision"] = summary["Avg Precision"].apply(lambda x: f"{x:.1%}")
    summary["Best Precision"] = summary["Best Precision"].apply(lambda x: f"{x:.1%}")

    print(tabulate(summary, headers="keys", tablefmt="simple", showindex=False))

    # Overall stats
    total_models = len(df)
    active_models = df["is_active"].sum()
    avg_precision = df["combined_precision"].mean()

    print(f"\nTotal Models: {total_models}")
    print(f"Active Models: {int(active_models)}")
    print(f"Average Precision: {avg_precision:.1%}")


def promote_model(engine, version: str, symbol: str, timeframe: str) -> bool:
    """Promote a specific model version to active."""
    with engine.begin() as conn:
        # Check if the model exists
        result = conn.execute(text("""
            SELECT id, combined_precision FROM model_versions
            WHERE version_name = :version AND symbol = :symbol AND timeframe = :timeframe
        """), {"version": version, "symbol": symbol, "timeframe": timeframe}).fetchone()

        if not result:
            print(f"Error: Model {version}/{symbol}/{timeframe} not found")
            return False

        model_id, precision = result

        # Deactivate current active model
        conn.execute(text("""
            UPDATE model_versions SET is_active = false
            WHERE symbol = :symbol AND timeframe = :timeframe AND is_active = true
        """), {"symbol": symbol, "timeframe": timeframe})

        # Activate the selected model
        conn.execute(text("""
            UPDATE model_versions SET is_active = true WHERE id = :id
        """), {"id": model_id})

        print(f"Promoted {version} to active for {symbol}/{timeframe}")
        print(f"Precision: {precision:.1%}")
        return True


def get_top_features(engine, version: str, symbol: str, timeframe: str) -> None:
    """Display top features for a model."""
    result = pd.read_sql(text("""
        SELECT top_features FROM model_versions
        WHERE version_name = :version AND symbol = :symbol AND timeframe = :timeframe
    """), engine, params={"version": version, "symbol": symbol, "timeframe": timeframe})

    if result.empty or result.iloc[0]["top_features"] is None:
        print("No feature importances found.")
        return

    features = result.iloc[0]["top_features"]

    print(f"\nTop Features for {version}/{symbol}/{timeframe}:")
    print("-" * 40)

    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_features:
        bar = "=" * int(importance * 100)
        print(f"  {name:25s} {importance:.4f} {bar}")


def main():
    parser = argparse.ArgumentParser(description="Compare model versions")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--timeframe", type=str, help="Filter by timeframe")
    parser.add_argument("--active", action="store_true", help="Show only active models")
    parser.add_argument("--compare", nargs=2, metavar=("SYMBOL", "TIMEFRAME"),
                        help="Compare versions for symbol/timeframe")
    parser.add_argument("--promote", nargs=3, metavar=("VERSION", "SYMBOL", "TIMEFRAME"),
                        help="Promote a model to active")
    parser.add_argument("--features", nargs=3, metavar=("VERSION", "SYMBOL", "TIMEFRAME"),
                        help="Show top features for a model")
    parser.add_argument("--summary", action="store_true", help="Show summary only")

    args = parser.parse_args()

    engine = create_engine(DB_URL)

    # Handle specific commands
    if args.promote:
        version, symbol, timeframe = args.promote
        promote_model(engine, version, symbol, timeframe)
        return

    if args.features:
        version, symbol, timeframe = args.features
        get_top_features(engine, version, symbol, timeframe)
        return

    # Get models
    df = get_models(engine, args.symbol, args.timeframe, args.active)

    if args.compare:
        symbol, timeframe = args.compare
        display_comparison(df, symbol, timeframe)
    elif args.summary:
        display_summary(df)
    else:
        # Show all models
        print(f"\n{'='*60}")
        print("MODEL VERSIONS")
        print(f"{'='*60}\n")
        display_models(df)
        display_summary(df)


if __name__ == "__main__":
    main()
