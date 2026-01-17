#!/usr/bin/env python3
"""
Test signal generation locally
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from ta import momentum, trend, volatility


# Database config - REQUIRED environment variables
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
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")

# Feature names
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

# ATR multipliers
ATR_SL_MULTIPLIER = 1.5
ATR_TP1_MULTIPLIER = 2.0
ATR_TP2_MULTIPLIER = 3.0
ATR_TP3_MULTIPLIER = 4.5


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features"""
    df = df.copy()
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]

    for period in [5, 15, 60, 240]:
        df[f"return_{period}"] = close.pct_change(period)
    df["log_return_60"] = np.log(close / close.shift(60))
    df["log_return_240"] = np.log(close / close.shift(240))

    for period in [7, 14]:
        atr = volatility.AverageTrueRange(high, low, close, window=period)
        df[f"atr_{period}"] = atr.average_true_range()
    df["atr_pct_14"] = df["atr_14"] / close * 100

    returns = close.pct_change()
    df["volatility_60"] = returns.rolling(60).std() * np.sqrt(60)
    df["volatility_240"] = returns.rolling(240).std() * np.sqrt(240)

    bb = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_width_20"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    bb_range = bb.bollinger_hband() - bb.bollinger_lband()
    df["bb_position_20"] = (close - bb.bollinger_lband()) / (bb_range + 1e-10)

    for period in [7, 14, 21]:
        rsi = momentum.RSIIndicator(close, window=period)
        df[f"rsi_{period}"] = rsi.rsi()

    macd_ind = trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()

    for period in [5, 10]:
        df[f"roc_{period}"] = momentum.ROCIndicator(close, window=period).roc()

    stoch = momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    adx = trend.ADXIndicator(high, low, close, window=14)
    df["adx_14"] = adx.adx()
    plus_di = adx.adx_pos()
    minus_di = adx.adx_neg()
    df["trend_strength"] = df["adx_14"] * np.sign(plus_di - minus_di) / 100

    ema8 = trend.EMAIndicator(close, window=8).ema_indicator()
    ema21 = trend.EMAIndicator(close, window=21).ema_indicator()
    ema55 = trend.EMAIndicator(close, window=55).ema_indicator()

    df["ema_ratio_8_21"] = ema8 / ema21 - 1
    df["ema_ratio_21_55"] = ema21 / ema55 - 1
    df["price_vs_ema_21"] = close / ema21 - 1
    df["price_vs_ema_55"] = close / ema55 - 1

    vol_sma = vol.rolling(20).mean()
    df["volume_sma_ratio_20"] = vol / (vol_sma + 1e-10)
    df["volume_momentum"] = vol.pct_change(5)

    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    df["distance_to_high_20"] = (high_20 - close) / close
    df["distance_to_low_20"] = (close - low_20) / close
    df["range_position"] = (close - low_20) / (high_20 - low_20 + 1e-10)

    return df


def find_latest_model(symbol: str, timeframe: str) -> Optional[str]:
    """Find latest model for symbol"""
    pattern = f"binary_{symbol}_{timeframe}_"
    matching = []

    for name in os.listdir(MODEL_DIR):
        if name.startswith(pattern):
            full_path = os.path.join(MODEL_DIR, name)
            if os.path.isdir(full_path):
                matching.append(full_path)

    if not matching:
        return None
    return sorted(matching)[-1]


def test_signal_generation():
    """Test signal generation for all symbols"""
    engine = create_engine(DB_URL)

    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
        "ALGOUSDT", "AAVEUSDT",
    ]

    timeframe = "4h"
    offset = "4h"
    signals = []

    print("=" * 70)
    print("QUANTUM TRADING AI - Signal Generation Test")
    print("=" * 70)

    for symbol in symbols:
        # Find model
        model_path = find_latest_model(symbol, timeframe)
        if not model_path:
            print(f"{symbol}: No model found")
            continue

        # Load model
        try:
            long_model = joblib.load(os.path.join(model_path, "long.joblib"))
            short_model = joblib.load(os.path.join(model_path, "short.joblib"))
            feature_cols = joblib.load(os.path.join(model_path, "features.joblib"))
            config = joblib.load(os.path.join(model_path, "config.joblib"))
        except Exception as e:
            print(f"{symbol}: Failed to load model - {e}")
            continue

        # Load data
        query = text("""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = :symbol AND timeframe = '1m'
            ORDER BY time DESC
            LIMIT 100000
        """)

        df = pd.read_sql(query, engine, params={"symbol": symbol}, parse_dates=["time"])
        df = df.sort_values("time").set_index("time")

        # Resample
        df = df.resample(offset).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

        if len(df) < 100:
            print(f"{symbol}: Not enough data")
            continue

        # Compute features
        df = compute_features(df)

        # Get latest features
        latest_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
        latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions
        long_proba = long_model.predict_proba(latest_features)[0, 1]
        short_proba = short_model.predict_proba(latest_features)[0, 1]

        long_threshold = config.get("long_threshold", 0.5)
        short_threshold = config.get("short_threshold", 0.5)

        # Determine signal
        direction = None
        confidence = 0

        if long_proba >= long_threshold and long_proba > short_proba:
            direction = "LONG"
            confidence = long_proba * 100
        elif short_proba >= short_threshold:
            direction = "SHORT"
            confidence = short_proba * 100

        # Get current price and ATR
        latest = df.iloc[-1]
        price = latest["close"]
        atr = latest["atr_14"]
        rsi = latest["rsi_14"]

        if direction:
            # Calculate levels
            if direction == "LONG":
                sl = price - atr * ATR_SL_MULTIPLIER
                tp1 = price + atr * ATR_TP1_MULTIPLIER
                tp2 = price + atr * ATR_TP2_MULTIPLIER
            else:
                sl = price + atr * ATR_SL_MULTIPLIER
                tp1 = price - atr * ATR_TP1_MULTIPLIER
                tp2 = price - atr * ATR_TP2_MULTIPLIER

            risk = abs(price - sl)
            reward = abs(tp2 - price)
            rr = reward / risk if risk > 0 else 0

            signals.append({
                "symbol": symbol,
                "direction": direction,
                "price": price,
                "confidence": confidence,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "rr": rr,
                "rsi": rsi,
            })

            emoji = "🟢" if direction == "LONG" else "🔴"
            print(f"{emoji} {symbol}: {direction} @ ${price:.4f} (conf: {confidence:.1f}%, RSI: {rsi:.1f}, R:R {rr:.1f})")
        else:
            print(f"⚪ {symbol}: No signal (L:{long_proba:.2f} S:{short_proba:.2f})")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(signals)} signals generated")
    print("=" * 70)

    if signals:
        print("\nFormatted signals for Telegram:\n")
        for sig in signals:
            emoji = "🟢" if sig["direction"] == "LONG" else "🔴"
            print(f"""
{emoji} *{sig['direction']} Signal* | {sig['symbol']}

📊 *Timeframe:* 4H
💰 *Entry:* ${sig['price']:.4f}
🎯 *Take Profit 1:* ${sig['tp1']:.4f}
🎯 *Take Profit 2:* ${sig['tp2']:.4f}
🛑 *Stop Loss:* ${sig['sl']:.4f}
📏 *R:R Ratio:* {sig['rr']:.1f}:1
🎯 *Confidence:* {sig['confidence']:.0f}%
📈 *RSI:* {sig['rsi']:.1f}

_⚠️ Not financial advice. Trade responsibly._
""")


if __name__ == "__main__":
    test_signal_generation()
