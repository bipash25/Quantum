"""
Quantum Trading AI - Model Training Script
==========================================
Trains ML models for signal generation.

Usage:
    python -m src.train --symbol BTCUSDT --timeframe 4h
    python -m src.train --all --timeframe 4h
"""
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from .config import settings
from .features import FeatureEngine, FeatureConfig
from .trainer import ModelTrainer, TrainingConfig, TrainingResult


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# MVP Symbols
MVP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
    "FTMUSDT", "ALGOUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT",
]

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def load_candle_data(
    symbol: str,
    days: int = 180,
    engine=None
) -> pd.DataFrame:
    """Load historical candle data from database"""
    if engine is None:
        engine = create_engine(settings.db_url)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    query = text("""
        SELECT time, open, high, low, close, volume, quote_volume
        FROM candles
        WHERE symbol = :symbol
          AND timeframe = '1m'
          AND time >= :start_date
          AND time <= :end_date
        ORDER BY time ASC
    """)

    df = pd.read_sql(
        query,
        engine,
        params={
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
        },
        parse_dates=["time"],
    )

    df.set_index("time", inplace=True)
    logger.info(f"Loaded {len(df)} candles for {symbol}")

    return df


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1-minute data to target timeframe"""
    if timeframe == "1m":
        return df

    # Map timeframe to pandas offset
    offset_map = {
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }

    offset = offset_map.get(timeframe, "1h")

    resampled = df.resample(offset).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
    }).dropna()

    logger.info(f"Resampled to {timeframe}: {len(resampled)} candles")
    return resampled


def train_model(
    symbol: str,
    timeframe: str,
    threshold_pct: float = 2.0,
) -> TrainingResult:
    """
    Train a model for a specific symbol and timeframe.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Signal timeframe ("4h" or "1d")
        threshold_pct: Price change threshold for signal classification

    Returns:
        TrainingResult with metrics and model paths
    """
    logger.info("=" * 60)
    logger.info(f"Training model for {symbol} {timeframe}")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading candle data...")
    df = load_candle_data(symbol, days=settings.training_days)

    if len(df) < settings.min_samples_for_training:
        raise ValueError(
            f"Not enough data for {symbol}: {len(df)} candles "
            f"(need {settings.min_samples_for_training})"
        )

    # Resample to target timeframe for feature computation
    # But keep 1-minute data for more granular features
    logger.info(f"Resampling to {timeframe}...")
    df_resampled = resample_to_timeframe(df.copy(), timeframe)

    # Compute features
    logger.info("Computing features...")
    feature_engine = FeatureEngine()
    df_features = feature_engine.compute_features(df_resampled)

    # Compute target
    timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 240)
    # For resampled data, adjust the forward looking period
    forward_periods = 1  # Look 1 bar ahead for the resampled timeframe

    logger.info(f"Computing target (threshold: {threshold_pct}%)...")
    target = feature_engine.compute_target(
        df_features,
        timeframe_minutes=forward_periods,  # 1 bar ahead
        threshold_pct=threshold_pct
    )

    # Get feature matrix
    X, feature_names = feature_engine.get_feature_matrix(df_features, dropna=False)
    y = target.values

    # Align X and y (remove NaN targets and features)
    valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Train model
    trainer = ModelTrainer()
    result = trainer.train(
        X=X,
        y=y,
        feature_names=feature_names,
        symbol=symbol,
        timeframe=timeframe,
    )

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model Version: {result.model_version}")
    logger.info(f"Accuracy: {result.metrics['accuracy']:.4f}")
    logger.info(f"Win Rate (Combined): {result.metrics['win_rate_combined']:.4f}")
    logger.info(f"Win Rate (Long): {result.metrics['win_rate_long']:.4f}")
    logger.info(f"Win Rate (Short): {result.metrics['win_rate_short']:.4f}")
    logger.info(f"Log Loss: {result.metrics['log_loss']:.4f}")
    logger.info("\nClassification Report:")
    logger.info(result.classification_report)
    logger.info("\nConfusion Matrix:")
    logger.info(result.confusion_matrix)
    logger.info("\nTop 10 Features:")
    for i, (name, imp) in enumerate(list(result.feature_importance.items())[:10]):
        logger.info(f"  {i+1}. {name}: {imp:.4f}")

    return result


def train_all_models(timeframe: str, symbols: List[str] = None) -> Dict[str, TrainingResult]:
    """Train models for all symbols"""
    if symbols is None:
        symbols = MVP_SYMBOLS

    results = {}

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Training {symbol}...")
        try:
            result = train_model(symbol, timeframe)
            results[symbol] = result
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    for symbol, result in results.items():
        logger.info(
            f"{symbol}: Accuracy={result.metrics['accuracy']:.3f}, "
            f"WinRate={result.metrics['win_rate_combined']:.3f}"
        )

    avg_accuracy = np.mean([r.metrics['accuracy'] for r in results.values()])
    avg_win_rate = np.mean([r.metrics['win_rate_combined'] for r in results.values()])

    logger.info(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average Win Rate: {avg_win_rate:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for trading signals"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Symbol to train (e.g., BTCUSDT)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train models for all MVP symbols"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="4h",
        choices=["1h", "4h", "1d"],
        help="Signal timeframe (default: 4h)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Price change threshold %% (default: 2.0)"
    )

    args = parser.parse_args()

    # Ensure model directory exists
    os.makedirs(settings.model_dir, exist_ok=True)

    if args.all:
        results = train_all_models(args.timeframe)
    elif args.symbol:
        result = train_model(args.symbol, args.timeframe, args.threshold)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
