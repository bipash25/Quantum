#!/usr/bin/env python3
"""
Quantum Trading AI - Training Script v3
========================================
Uses binary classifiers for LONG and SHORT signals separately.

This approach works better for imbalanced data:
- Train one model to detect LONG opportunities (return > threshold)
- Train one model to detect SHORT opportunities (return < -threshold)
- Generate signal when classifier predicts positive with high confidence

Usage:
    python scripts/train_model.py --symbol BTCUSDT --timeframe 4h
    python scripts/train_model.py --all --timeframe 4h
"""
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from ta import momentum, trend, volatility, volume
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Database config - REQUIRED environment variables
DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    # Construct from individual env vars for local development
    db_user = os.environ.get("DB_USER", "")
    db_pass = os.environ.get("DB_PASSWORD", "")
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "quantum_trading")
    if db_user and db_pass:
        DB_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    else:
        raise ValueError("DATABASE_URL or DB_USER/DB_PASSWORD environment variables required")

# Model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# MVP Symbols (excluding those with insufficient data)
MVP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
    "ALGOUSDT", "AAVEUSDT",
]


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

FEATURE_NAMES = [
    "return_5", "return_15", "return_60", "return_240",
    "log_return_60", "log_return_240",
    "atr_7", "atr_14", "atr_pct_14",
    "volatility_60", "volatility_240",
    "bb_width_20", "bb_position_20",
    "rsi_7", "rsi_14", "rsi_21",
    "macd", "macd_signal", "macd_diff",
    "roc_5", "roc_10",
    "stoch_k", "stoch_d",
    "adx_14",
    "ema_ratio_8_21", "ema_ratio_21_55",
    "price_vs_ema_21", "price_vs_ema_55",
    "trend_strength",
    "volume_sma_ratio_20", "volume_momentum",
    "distance_to_high_20", "distance_to_low_20",
    "range_position",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicator features"""
    df = df.copy()
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]

    # Returns
    for period in [5, 15, 60, 240]:
        df[f"return_{period}"] = close.pct_change(period)
    df["log_return_60"] = np.log(close / close.shift(60))
    df["log_return_240"] = np.log(close / close.shift(240))

    # Volatility - ATR
    for period in [7, 14]:
        atr = volatility.AverageTrueRange(high, low, close, window=period)
        df[f"atr_{period}"] = atr.average_true_range()
    df["atr_pct_14"] = df["atr_14"] / close * 100

    # Rolling volatility
    returns = close.pct_change()
    df["volatility_60"] = returns.rolling(60).std() * np.sqrt(60)
    df["volatility_240"] = returns.rolling(240).std() * np.sqrt(240)

    # Bollinger Bands
    bb = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_width_20"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    bb_range = bb.bollinger_hband() - bb.bollinger_lband()
    df["bb_position_20"] = (close - bb.bollinger_lband()) / (bb_range + 1e-10)

    # RSI
    for period in [7, 14, 21]:
        rsi = momentum.RSIIndicator(close, window=period)
        df[f"rsi_{period}"] = rsi.rsi()

    # MACD
    macd = trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # ROC
    for period in [5, 10]:
        df[f"roc_{period}"] = momentum.ROCIndicator(close, window=period).roc()

    # Stochastic
    stoch = momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ADX
    adx = trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"] = adx.adx()
    plus_di = adx.adx_pos()
    minus_di = adx.adx_neg()
    df["trend_strength"] = df["adx_14"] * np.sign(plus_di - minus_di) / 100

    # EMAs
    ema8 = trend.EMAIndicator(close, window=8).ema_indicator()
    ema21 = trend.EMAIndicator(close, window=21).ema_indicator()
    ema55 = trend.EMAIndicator(close, window=55).ema_indicator()

    df["ema_ratio_8_21"] = ema8 / ema21 - 1
    df["ema_ratio_21_55"] = ema21 / ema55 - 1
    df["price_vs_ema_21"] = close / ema21 - 1
    df["price_vs_ema_55"] = close / ema55 - 1

    # Volume
    vol_sma = vol.rolling(20).mean()
    df["volume_sma_ratio_20"] = vol / (vol_sma + 1e-10)
    df["volume_momentum"] = vol.pct_change(5)

    # Market structure
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    df["distance_to_high_20"] = (high_20 - close) / close
    df["distance_to_low_20"] = (close - low_20) / close
    df["range_position"] = (close - low_20) / (high_20 - low_20 + 1e-10)

    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """Load candle data from database"""
    engine = create_engine(DB_URL)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    query = text("""
        SELECT time, open, high, low, close, volume
        FROM candles
        WHERE symbol = :symbol AND timeframe = '1m'
          AND time >= :start_date AND time <= :end_date
        ORDER BY time ASC
    """)

    df = pd.read_sql(query, engine, params={
        "symbol": symbol, "start_date": start_date, "end_date": end_date
    }, parse_dates=["time"])

    df.set_index("time", inplace=True)
    logger.info(f"Loaded {len(df)} candles for {symbol}")
    return df


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample to target timeframe"""
    offset = {"1h": "1h", "4h": "4h", "1d": "1D"}.get(timeframe, "4h")

    resampled = df.resample(offset).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    logger.info(f"Resampled to {timeframe}: {len(resampled)} bars")
    return resampled


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_lgbm_params(is_long: bool = True) -> Dict:
    """Get LightGBM parameters for binary classification"""
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.02,
        "n_estimators": 600,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "scale_pos_weight": 10,  # Weight positive class more heavily
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                           min_signals: int = 3) -> Tuple[float, Dict]:
    """Find optimal probability threshold for best precision"""
    best_threshold = 0.5
    best_score = 0
    best_metrics = {}

    for thresh in np.arange(0.3, 0.85, 0.05):
        preds = (y_proba >= thresh).astype(int)
        n_positive = preds.sum()

        if n_positive < min_signals:
            continue

        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        # Score: prioritize precision but require some signals
        score = precision * 0.7 + (min(n_positive, 15) / 15) * 0.3

        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                "threshold": thresh,
                "n_signals": int(n_positive),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    return best_threshold, best_metrics


def train_model(symbol: str, timeframe: str, threshold_pct: float = 1.5) -> Dict[str, Any]:
    """Train binary classifiers for LONG and SHORT signals"""
    logger.info("=" * 60)
    logger.info(f"Training {symbol} {timeframe} (threshold: {threshold_pct}%)")
    logger.info("=" * 60)

    # Load and prepare data
    df_raw = load_data(symbol, days=180)
    if len(df_raw) < 50000:
        raise ValueError(f"Not enough data: {len(df_raw)} candles")

    df = resample_data(df_raw, timeframe)
    min_bars = 300 if timeframe == "1d" else 500
    if len(df) < min_bars:
        raise ValueError(f"Not enough resampled data: {len(df)} bars (need {min_bars})")

    # Compute features
    logger.info("Computing features...")
    df = compute_features(df)

    # Compute forward returns
    forward_return = df["close"].shift(-1) / df["close"] - 1
    forward_return_pct = forward_return * 100

    # Binary targets
    y_long = (forward_return_pct > threshold_pct).astype(int)
    y_short = (forward_return_pct < -threshold_pct).astype(int)

    # Prepare feature matrix
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].values

    # Remove NaN rows
    valid = ~np.isnan(forward_return) & ~np.isnan(X).any(axis=1)
    X = X[valid]
    y_long = y_long.values[valid]
    y_short = y_short.values[valid]

    # Replace infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"LONG signals: {y_long.sum()} ({y_long.mean()*100:.1f}%)")
    logger.info(f"SHORT signals: {y_short.sum()} ({y_short.mean()*100:.1f}%)")

    # Train/validation split (80/20, time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_long_train, y_long_val = y_long[:split_idx], y_long[split_idx:]
    y_short_train, y_short_val = y_short[:split_idx], y_short[split_idx:]

    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    logger.info(f"Val LONG: {y_long_val.sum()}, Val SHORT: {y_short_val.sum()}")

    # =========================================================================
    # LONG Classifier
    # =========================================================================
    logger.info("\n--- Training LONG Classifier ---")
    long_model = lgb.LGBMClassifier(**get_lgbm_params(is_long=True))
    long_weights = compute_sample_weight("balanced", y_long_train)
    long_model.fit(X_train, y_long_train, sample_weight=long_weights)

    long_proba_val = long_model.predict_proba(X_val)[:, 1]
    long_threshold, long_metrics = find_optimal_threshold(y_long_val, long_proba_val)

    if long_metrics:
        logger.info(f"LONG - Threshold: {long_threshold:.2f}")
        logger.info(f"LONG - Signals: {long_metrics['n_signals']}, Precision: {long_metrics['precision']:.3f}")
    else:
        long_threshold = 0.5
        logger.info("LONG - No optimal threshold found, using 0.5")

    # =========================================================================
    # SHORT Classifier
    # =========================================================================
    logger.info("\n--- Training SHORT Classifier ---")
    short_model = lgb.LGBMClassifier(**get_lgbm_params(is_long=False))
    short_weights = compute_sample_weight("balanced", y_short_train)
    short_model.fit(X_train, y_short_train, sample_weight=short_weights)

    short_proba_val = short_model.predict_proba(X_val)[:, 1]
    short_threshold, short_metrics = find_optimal_threshold(y_short_val, short_proba_val)

    if short_metrics:
        logger.info(f"SHORT - Threshold: {short_threshold:.2f}")
        logger.info(f"SHORT - Signals: {short_metrics['n_signals']}, Precision: {short_metrics['precision']:.3f}")
    else:
        short_threshold = 0.5
        logger.info("SHORT - No optimal threshold found, using 0.5")

    # =========================================================================
    # Combined Evaluation
    # =========================================================================
    logger.info("\n--- Combined Signal Evaluation ---")

    # Generate predictions using optimal thresholds
    long_preds = (long_proba_val >= long_threshold).astype(int)
    short_preds = (short_proba_val >= short_threshold).astype(int)

    # Win rates
    n_long = long_preds.sum()
    n_short = short_preds.sum()

    if n_long > 0:
        long_precision = (y_long_val[long_preds == 1] == 1).mean()
    else:
        long_precision = 0.0

    if n_short > 0:
        short_precision = (y_short_val[short_preds == 1] == 1).mean()
    else:
        short_precision = 0.0

    n_signals = n_long + n_short
    if n_signals > 0:
        combined_precision = (n_long * long_precision + n_short * short_precision) / n_signals
    else:
        combined_precision = 0.0

    logger.info(f"LONG:  {n_long} signals, {long_precision:.1%} win rate")
    logger.info(f"SHORT: {n_short} signals, {short_precision:.1%} win rate")
    logger.info(f"TOTAL: {n_signals} signals, {combined_precision:.1%} combined win rate")

    # AUC scores
    try:
        long_auc = roc_auc_score(y_long_val, long_proba_val)
        short_auc = roc_auc_score(y_short_val, short_proba_val)
        logger.info(f"LONG AUC: {long_auc:.3f}, SHORT AUC: {short_auc:.3f}")
    except:
        long_auc = short_auc = 0.5

    # =========================================================================
    # Train final models on all data
    # =========================================================================
    logger.info("\nTraining final models on all data...")

    long_model_final = lgb.LGBMClassifier(**get_lgbm_params(is_long=True))
    long_weights_all = compute_sample_weight("balanced", y_long)
    long_model_final.fit(X, y_long, sample_weight=long_weights_all)

    short_model_final = lgb.LGBMClassifier(**get_lgbm_params(is_long=False))
    short_weights_all = compute_sample_weight("balanced", y_short)
    short_model_final.fit(X, y_short, sample_weight=short_weights_all)

    # =========================================================================
    # Save models
    # =========================================================================
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = f"binary_{symbol}_{timeframe}_{version}"
    model_path = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_path, exist_ok=True)

    model_config = {
        "symbol": symbol,
        "timeframe": timeframe,
        "threshold_pct": threshold_pct,
        "long_threshold": float(long_threshold),
        "short_threshold": float(short_threshold),
        "train_date": version,
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "long_precision": float(long_precision),
        "short_precision": float(short_precision),
        "combined_precision": float(combined_precision),
        "n_signals": int(n_signals),
        "long_auc": float(long_auc),
        "short_auc": float(short_auc),
    }

    joblib.dump(long_model_final, os.path.join(model_path, "long.joblib"))
    joblib.dump(short_model_final, os.path.join(model_path, "short.joblib"))
    joblib.dump(feature_cols, os.path.join(model_path, "features.joblib"))
    joblib.dump(model_config, os.path.join(model_path, "config.joblib"))

    logger.info(f"Models saved to {model_path}")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "long_precision": long_precision,
        "short_precision": short_precision,
        "combined_precision": combined_precision,
        "n_long": n_long,
        "n_short": n_short,
        "n_signals": n_signals,
        "long_auc": long_auc,
        "short_auc": short_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--symbol", type=str, help="Symbol to train")
    parser.add_argument("--all", action="store_true", help="Train all symbols")
    parser.add_argument("--timeframe", type=str, default="4h", choices=["1h", "4h", "1d"])
    parser.add_argument("--threshold", type=float, default=1.5, help="Price threshold %% (default: 1.5)")

    args = parser.parse_args()

    if args.all:
        results = {}
        for symbol in MVP_SYMBOLS:
            try:
                result = train_model(symbol, args.timeframe, args.threshold)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Failed {symbol}: {e}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        for symbol, result in results.items():
            logger.info(
                f"{symbol}: WinRate={result['combined_precision']:.1%}, "
                f"Signals={result['n_signals']} (L:{result['n_long']}/S:{result['n_short']})"
            )

        if results:
            valid_results = [r for r in results.values() if r['n_signals'] > 0]
            if valid_results:
                avg_wr = np.mean([r['combined_precision'] for r in valid_results])
                total_signals = sum([r['n_signals'] for r in valid_results])
                avg_auc = np.mean([(r['long_auc'] + r['short_auc']) / 2 for r in valid_results])
                logger.info(f"\nAverage Win Rate: {avg_wr:.1%}")
                logger.info(f"Average AUC: {avg_auc:.3f}")
                logger.info(f"Total Signals in Validation: {total_signals}")

    elif args.symbol:
        train_model(args.symbol, args.timeframe, args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
