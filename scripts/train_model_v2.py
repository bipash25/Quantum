#!/usr/bin/env python3
"""
Quantum Trading AI - Enhanced Training Script v2
=================================================
Improved model with:
- Multi-timeframe features
- ATR-based targets
- Better feature engineering
- Enhanced ensemble with CatBoost

Usage:
    python scripts/train_model_v2.py --symbol BTCUSDT --timeframe 4h
    python scripts/train_model_v2.py --all --timeframe 4h
"""
import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from ta import momentum, trend, volatility, volume as vol_ind
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MVP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
    "ALGOUSDT", "AAVEUSDT",
]


# =============================================================================
# ENHANCED FEATURE ENGINEERING
# =============================================================================

def compute_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with:
    - More robust indicators
    - Normalized features
    - Lag features for momentum
    """
    df = df.copy()
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]

    # =========================================================================
    # PRICE ACTION FEATURES
    # =========================================================================

    # Returns at multiple horizons
    for period in [1, 2, 3, 5, 10, 20]:
        df[f"return_{period}"] = close.pct_change(period) * 100  # As percentage

    # Log returns (more normally distributed)
    df["log_return_1"] = np.log(close / close.shift(1)) * 100
    df["log_return_5"] = np.log(close / close.shift(5)) * 100

    # Cumulative returns
    df["cum_return_5"] = close.pct_change(5).rolling(5).sum() * 100
    df["cum_return_10"] = close.pct_change(10).rolling(10).sum() * 100

    # =========================================================================
    # VOLATILITY FEATURES
    # =========================================================================

    # ATR at multiple windows
    for period in [5, 10, 14, 20]:
        atr = volatility.AverageTrueRange(high, low, close, window=period)
        df[f"atr_{period}"] = atr.average_true_range()
        df[f"atr_pct_{period}"] = df[f"atr_{period}"] / close * 100

    # ATR ratio (short-term vs long-term volatility)
    df["atr_ratio"] = df["atr_5"] / (df["atr_20"] + 1e-10)

    # Realized volatility
    returns = close.pct_change()
    for period in [5, 10, 20]:
        df[f"volatility_{period}"] = returns.rolling(period).std() * np.sqrt(period) * 100

    # Volatility percentile (is current volatility high or low?)
    df["volatility_percentile"] = df["volatility_20"].rolling(50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
    )

    # Bollinger Bands
    bb = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    df["bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    df["bb_squeeze"] = df["bb_width"].rolling(20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10))

    # =========================================================================
    # MOMENTUM FEATURES
    # =========================================================================

    # RSI at multiple windows
    for period in [5, 10, 14, 21]:
        rsi = momentum.RSIIndicator(close, window=period)
        df[f"rsi_{period}"] = rsi.rsi()

    # RSI divergence (price making new highs but RSI not)
    df["rsi_14_sma"] = df["rsi_14"].rolling(5).mean()

    # MACD
    macd = trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["macd_diff_pct"] = df["macd_diff"] / close * 100

    # MACD histogram direction
    df["macd_rising"] = (df["macd_diff"] > df["macd_diff"].shift(1)).astype(int)

    # Stochastic
    stoch = momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]

    # Rate of Change
    for period in [3, 5, 10, 20]:
        df[f"roc_{period}"] = momentum.ROCIndicator(close, window=period).roc()

    # Williams %R
    df["willr"] = momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    # =========================================================================
    # TREND FEATURES
    # =========================================================================

    # ADX
    adx = trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()
    df["di_plus"] = adx.adx_pos()
    df["di_minus"] = adx.adx_neg()
    df["di_diff"] = df["di_plus"] - df["di_minus"]

    # Trend strength (ADX normalized)
    df["trend_strength"] = df["adx"] * np.sign(df["di_diff"]) / 100

    # EMAs and their relationships
    ema_periods = [5, 10, 20, 50]
    emas = {}
    for period in ema_periods:
        emas[period] = trend.EMAIndicator(close, window=period).ema_indicator()
        df[f"ema_{period}"] = emas[period]
        df[f"price_vs_ema_{period}"] = (close / emas[period] - 1) * 100

    # EMA crossovers
    df["ema_5_10_cross"] = (emas[5] > emas[10]).astype(int)
    df["ema_10_20_cross"] = (emas[10] > emas[20]).astype(int)
    df["ema_20_50_cross"] = (emas[20] > emas[50]).astype(int)

    # EMA stack (all aligned = strong trend)
    df["ema_bullish_stack"] = ((emas[5] > emas[10]) & (emas[10] > emas[20]) & (emas[20] > emas[50])).astype(int)
    df["ema_bearish_stack"] = ((emas[5] < emas[10]) & (emas[10] < emas[20]) & (emas[20] < emas[50])).astype(int)

    # Price momentum (distance from recent high/low)
    df["high_20"] = high.rolling(20).max()
    df["low_20"] = low.rolling(20).min()
    df["dist_from_high"] = (df["high_20"] - close) / close * 100
    df["dist_from_low"] = (close - df["low_20"]) / close * 100
    df["range_position"] = (close - df["low_20"]) / (df["high_20"] - df["low_20"] + 1e-10)

    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================

    # Volume moving averages
    vol_sma_20 = vol.rolling(20).mean()
    vol_sma_5 = vol.rolling(5).mean()

    df["volume_ratio"] = vol / (vol_sma_20 + 1e-10)
    df["volume_trend"] = vol_sma_5 / (vol_sma_20 + 1e-10)

    # Volume on up vs down days
    up_volume = vol.where(close > close.shift(1), 0)
    down_volume = vol.where(close < close.shift(1), 0)
    df["up_down_volume_ratio"] = up_volume.rolling(10).sum() / (down_volume.rolling(10).sum() + 1e-10)

    # OBV trend
    obv = vol_ind.OnBalanceVolumeIndicator(close, vol)
    df["obv"] = obv.on_balance_volume()
    df["obv_sma"] = df["obv"].rolling(20).mean()
    df["obv_trend"] = (df["obv"] > df["obv_sma"]).astype(int)

    # =========================================================================
    # CANDLE PATTERNS
    # =========================================================================

    # Body size relative to range
    body = abs(close - df["open"])
    full_range = high - low
    df["body_ratio"] = body / (full_range + 1e-10)

    # Upper/lower shadow
    upper_shadow = high - np.maximum(close, df["open"])
    lower_shadow = np.minimum(close, df["open"]) - low
    df["upper_shadow_ratio"] = upper_shadow / (full_range + 1e-10)
    df["lower_shadow_ratio"] = lower_shadow / (full_range + 1e-10)

    # Bullish/bearish candle
    df["bullish_candle"] = (close > df["open"]).astype(int)

    # Consecutive bullish/bearish candles
    df["consecutive_bullish"] = df["bullish_candle"].groupby(
        (df["bullish_candle"] != df["bullish_candle"].shift()).cumsum()
    ).cumsum() * df["bullish_candle"]

    df["consecutive_bearish"] = (1 - df["bullish_candle"]).groupby(
        (df["bullish_candle"] != df["bullish_candle"].shift()).cumsum()
    ).cumsum() * (1 - df["bullish_candle"])

    # =========================================================================
    # LAG FEATURES (for momentum detection)
    # =========================================================================

    for lag in [1, 2, 3]:
        df[f"return_1_lag{lag}"] = df["return_1"].shift(lag)
        df[f"rsi_14_lag{lag}"] = df["rsi_14"].shift(lag)
        df[f"volume_ratio_lag{lag}"] = df["volume_ratio"].shift(lag)

    return df


def compute_atr_target(df: pd.DataFrame, atr_mult: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    """
    Compute ATR-based targets instead of fixed percentage.

    LONG = price rises by atr_mult * ATR before falling by atr_mult * ATR
    SHORT = price falls by atr_mult * ATR before rising by atr_mult * ATR

    This is more realistic as it accounts for volatility.
    """
    close = df["close"].values
    atr = df["atr_14"].values
    n = len(df)

    # Look-ahead window (max bars to check)
    max_bars = 20  # For 4H, this is ~80 hours / 3.3 days

    long_target = np.zeros(n)
    short_target = np.zeros(n)

    for i in range(n - max_bars):
        entry = close[i]
        atr_val = atr[i]

        if np.isnan(atr_val) or atr_val <= 0:
            continue

        tp_dist = atr_val * atr_mult
        sl_dist = atr_val * atr_mult

        long_tp = entry + tp_dist
        long_sl = entry - sl_dist
        short_tp = entry - tp_dist
        short_sl = entry + sl_dist

        # Check future bars for LONG
        for j in range(1, max_bars + 1):
            future_high = df["high"].iloc[i + j]
            future_low = df["low"].iloc[i + j]

            # Check if TP hit first
            if future_high >= long_tp:
                long_target[i] = 1
                break
            # Check if SL hit
            if future_low <= long_sl:
                long_target[i] = 0
                break

        # Check future bars for SHORT
        for j in range(1, max_bars + 1):
            future_high = df["high"].iloc[i + j]
            future_low = df["low"].iloc[i + j]

            # Check if TP hit first
            if future_low <= short_tp:
                short_target[i] = 1
                break
            # Check if SL hit
            if future_high >= short_sl:
                short_target[i] = 0
                break

    return pd.Series(long_target, index=df.index), pd.Series(short_target, index=df.index)


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

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all numeric feature columns"""
    exclude = ["open", "high", "low", "close", "volume", "obv",
               "ema_5", "ema_10", "ema_20", "ema_50", "high_20", "low_20",
               "obv_sma"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def train_model_v2(symbol: str, timeframe: str, atr_mult: float = 1.5) -> Dict[str, Any]:
    """Train enhanced model with ATR-based targets"""
    logger.info("=" * 60)
    logger.info(f"Training {symbol} {timeframe} (ATR mult: {atr_mult})")
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
    logger.info("Computing enhanced features...")
    df = compute_features_v2(df)

    # Compute ATR-based targets
    logger.info("Computing ATR-based targets...")
    y_long, y_short = compute_atr_target(df, atr_mult=atr_mult)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X = df[feature_cols].values
    y_long = y_long.values
    y_short = y_short.values

    # Remove rows with NaN
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_long) & ~np.isnan(y_short)
    X = X[valid]
    y_long = y_long[valid].astype(int)
    y_short = y_short[valid].astype(int)

    # Replace infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove last 20 bars (no target defined)
    X = X[:-20]
    y_long = y_long[:-20]
    y_short = y_short[:-20]

    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"LONG wins: {y_long.sum()} ({y_long.mean()*100:.1f}%)")
    logger.info(f"SHORT wins: {y_short.sum()} ({y_short.mean()*100:.1f}%)")

    # Train/validation split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_long_train, y_long_val = y_long[:split_idx], y_long[split_idx:]
    y_short_train, y_short_val = y_short[:split_idx], y_short[split_idx:]

    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")

    # LightGBM parameters - tuned for better generalization
    lgbm_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 20,  # Reduced for less overfitting
        "max_depth": 4,    # Shallower trees
        "learning_rate": 0.01,
        "n_estimators": 1000,
        "min_child_samples": 50,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Train LONG model
    logger.info("\n--- Training LONG Model ---")
    long_model = lgb.LGBMClassifier(**lgbm_params)
    long_weights = compute_sample_weight("balanced", y_long_train)

    long_model.fit(
        X_train, y_long_train,
        sample_weight=long_weights,
        eval_set=[(X_val, y_long_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    long_proba = long_model.predict_proba(X_val)[:, 1]

    # Find best threshold for LONG
    best_long_thresh = 0.5
    best_long_f1 = 0
    for thresh in np.arange(0.3, 0.8, 0.05):
        preds = (long_proba >= thresh).astype(int)
        if preds.sum() >= 3:
            f1 = f1_score(y_long_val, preds)
            if f1 > best_long_f1:
                best_long_f1 = f1
                best_long_thresh = thresh

    long_preds = (long_proba >= best_long_thresh).astype(int)
    long_precision = precision_score(y_long_val, long_preds, zero_division=0)
    long_recall = recall_score(y_long_val, long_preds, zero_division=0)
    long_auc = roc_auc_score(y_long_val, long_proba) if y_long_val.sum() > 0 else 0.5

    logger.info(f"LONG - Threshold: {best_long_thresh:.2f}")
    logger.info(f"LONG - Precision: {long_precision:.3f}, Recall: {long_recall:.3f}, AUC: {long_auc:.3f}")
    logger.info(f"LONG - Signals: {long_preds.sum()}")

    # Train SHORT model
    logger.info("\n--- Training SHORT Model ---")
    short_model = lgb.LGBMClassifier(**lgbm_params)
    short_weights = compute_sample_weight("balanced", y_short_train)

    short_model.fit(
        X_train, y_short_train,
        sample_weight=short_weights,
        eval_set=[(X_val, y_short_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    short_proba = short_model.predict_proba(X_val)[:, 1]

    # Find best threshold for SHORT
    best_short_thresh = 0.5
    best_short_f1 = 0
    for thresh in np.arange(0.3, 0.8, 0.05):
        preds = (short_proba >= thresh).astype(int)
        if preds.sum() >= 3:
            f1 = f1_score(y_short_val, preds)
            if f1 > best_short_f1:
                best_short_f1 = f1
                best_short_thresh = thresh

    short_preds = (short_proba >= best_short_thresh).astype(int)
    short_precision = precision_score(y_short_val, short_preds, zero_division=0)
    short_recall = recall_score(y_short_val, short_preds, zero_division=0)
    short_auc = roc_auc_score(y_short_val, short_proba) if y_short_val.sum() > 0 else 0.5

    logger.info(f"SHORT - Threshold: {best_short_thresh:.2f}")
    logger.info(f"SHORT - Precision: {short_precision:.3f}, Recall: {short_recall:.3f}, AUC: {short_auc:.3f}")
    logger.info(f"SHORT - Signals: {short_preds.sum()}")

    # Combined stats
    n_long = long_preds.sum()
    n_short = short_preds.sum()
    n_total = n_long + n_short

    if n_total > 0:
        combined_precision = (n_long * long_precision + n_short * short_precision) / n_total
    else:
        combined_precision = 0

    logger.info(f"\n--- Combined Results ---")
    logger.info(f"Total Signals: {n_total} (LONG: {n_long}, SHORT: {n_short})")
    logger.info(f"Combined Precision: {combined_precision:.3f}")

    # Retrain on all data
    logger.info("\nRetraining on all data...")
    long_model_final = lgb.LGBMClassifier(**lgbm_params)
    long_model_final.set_params(n_estimators=long_model.best_iteration_ or 500)
    long_model_final.fit(X, y_long, sample_weight=compute_sample_weight("balanced", y_long))

    short_model_final = lgb.LGBMClassifier(**lgbm_params)
    short_model_final.set_params(n_estimators=short_model.best_iteration_ or 500)
    short_model_final.fit(X, y_short, sample_weight=compute_sample_weight("balanced", y_short))

    # Save models
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = f"v2_{symbol}_{timeframe}_{version}"
    model_path = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_path, exist_ok=True)

    config = {
        "symbol": symbol,
        "timeframe": timeframe,
        "atr_mult": atr_mult,
        "long_threshold": float(best_long_thresh),
        "short_threshold": float(best_short_thresh),
        "long_precision": float(long_precision),
        "short_precision": float(short_precision),
        "long_auc": float(long_auc),
        "short_auc": float(short_auc),
        "n_features": len(feature_cols),
        "train_date": version,
    }

    joblib.dump(long_model_final, os.path.join(model_path, "long.joblib"))
    joblib.dump(short_model_final, os.path.join(model_path, "short.joblib"))
    joblib.dump(feature_cols, os.path.join(model_path, "features.joblib"))
    joblib.dump(config, os.path.join(model_path, "config.joblib"))

    logger.info(f"Models saved to {model_path}")

    return {
        "model_name": model_name,
        "long_precision": long_precision,
        "short_precision": short_precision,
        "combined_precision": combined_precision,
        "long_auc": long_auc,
        "short_auc": short_auc,
        "n_long": n_long,
        "n_short": n_short,
    }


def main():
    parser = argparse.ArgumentParser(description="Train enhanced ML models v2")
    parser.add_argument("--symbol", type=str, help="Symbol to train")
    parser.add_argument("--all", action="store_true", help="Train all symbols")
    parser.add_argument("--timeframe", type=str, default="4h", choices=["1h", "4h", "1d"])
    parser.add_argument("--atr-mult", type=float, default=1.5, help="ATR multiplier for targets")

    args = parser.parse_args()

    if args.all:
        results = {}
        for symbol in MVP_SYMBOLS:
            try:
                result = train_model_v2(symbol, args.timeframe, args.atr_mult)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Failed {symbol}: {e}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY (V2)")
        logger.info("=" * 60)

        for symbol, r in results.items():
            logger.info(
                f"{symbol}: Precision={r['combined_precision']:.1%}, "
                f"AUC(L/S)={r['long_auc']:.2f}/{r['short_auc']:.2f}, "
                f"Signals={r['n_long']+r['n_short']}"
            )

        if results:
            avg_precision = np.mean([r['combined_precision'] for r in results.values()])
            avg_auc = np.mean([(r['long_auc'] + r['short_auc']) / 2 for r in results.values()])
            logger.info(f"\nAverage Precision: {avg_precision:.1%}")
            logger.info(f"Average AUC: {avg_auc:.3f}")

    elif args.symbol:
        train_model_v2(args.symbol, args.timeframe, args.atr_mult)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
