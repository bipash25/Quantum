"""
Quantum Trading AI - Scheduled Signal Generator
================================================
Runs signal generation on a 4-hour and 24-hour schedule aligned with candle closes.

Schedule:
- 4H candles close at: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
- 24H candles close at: 00:00 UTC
- We run signals 5 minutes after each close to ensure data is complete
"""
import asyncio
import logging
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

import aiohttp
import numpy as np
import pandas as pd
import redis
import joblib
from sqlalchemy import create_engine, text
from ta import momentum, trend, volatility, volume as vol_ind
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import settings


logger = logging.getLogger(__name__)


# Feature names (must match training)
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

# Top 50 Symbols to trade (by market cap)
SYMBOLS = [
    # Top 20 (original MVP)
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
    "FTMUSDT", "ALGOUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT",

    # Additional 30 (top 50 total)
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT",
    "TIAUSDT", "SEIUSDT", "RUNEUSDT", "RENDERUSDT", "WLDUSDT",
    "IMXUSDT", "LDOUSDT", "STXUSDT", "FILUSDT", "HBARUSDT",
    "VETUSDT", "ICPUSDT", "MKRUSDT", "QNTUSDT", "GRTUSDT",
    "FLOWUSDT", "XLMUSDT", "AXSUSDT", "THETAUSDT", "EGLDUSDT",
    "APEUSDT", "CHZUSDT", "EOSUSDT", "CFXUSDT", "ZILUSDT",
]


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


def compute_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering for v2 models"""
    df = df.copy()
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]

    # Returns at multiple horizons
    for period in [1, 2, 3, 5, 10, 20]:
        df[f"return_{period}"] = close.pct_change(period) * 100

    # Log returns
    df["log_return_1"] = np.log(close / close.shift(1)) * 100
    df["log_return_5"] = np.log(close / close.shift(5)) * 100

    # Cumulative returns
    df["cum_return_5"] = close.pct_change(5).rolling(5).sum() * 100
    df["cum_return_10"] = close.pct_change(10).rolling(10).sum() * 100

    # ATR at multiple windows
    for period in [5, 10, 14, 20]:
        atr = volatility.AverageTrueRange(high, low, close, window=period)
        df[f"atr_{period}"] = atr.average_true_range()
        df[f"atr_pct_{period}"] = df[f"atr_{period}"] / close * 100

    df["atr_ratio"] = df["atr_5"] / (df["atr_20"] + 1e-10)

    # Realized volatility
    returns = close.pct_change()
    for period in [5, 10, 20]:
        df[f"volatility_{period}"] = returns.rolling(period).std() * np.sqrt(period) * 100

    df["volatility_percentile"] = df["volatility_20"].rolling(50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
    )

    # Bollinger Bands
    bb = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    df["bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    df["bb_squeeze"] = df["bb_width"].rolling(20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10))

    # RSI at multiple windows
    for period in [5, 10, 14, 21]:
        rsi = momentum.RSIIndicator(close, window=period)
        df[f"rsi_{period}"] = rsi.rsi()

    df["rsi_14_sma"] = df["rsi_14"].rolling(5).mean()

    # MACD
    macd = trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    df["macd_diff_pct"] = df["macd_diff"] / close * 100
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

    # ADX
    adx = trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()
    df["di_plus"] = adx.adx_pos()
    df["di_minus"] = adx.adx_neg()
    df["di_diff"] = df["di_plus"] - df["di_minus"]
    df["trend_strength"] = df["adx"] * np.sign(df["di_diff"]) / 100

    # EMAs
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

    # EMA stack
    df["ema_bullish_stack"] = ((emas[5] > emas[10]) & (emas[10] > emas[20]) & (emas[20] > emas[50])).astype(int)
    df["ema_bearish_stack"] = ((emas[5] < emas[10]) & (emas[10] < emas[20]) & (emas[20] < emas[50])).astype(int)

    # Price momentum
    df["high_20"] = high.rolling(20).max()
    df["low_20"] = low.rolling(20).min()
    df["dist_from_high"] = (df["high_20"] - close) / close * 100
    df["dist_from_low"] = (close - df["low_20"]) / close * 100
    df["range_position"] = (close - df["low_20"]) / (df["high_20"] - df["low_20"] + 1e-10)

    # Volume features
    vol_sma_20 = vol.rolling(20).mean()
    vol_sma_5 = vol.rolling(5).mean()
    df["volume_ratio"] = vol / (vol_sma_20 + 1e-10)
    df["volume_trend"] = vol_sma_5 / (vol_sma_20 + 1e-10)

    up_volume = vol.where(close > close.shift(1), 0)
    down_volume = vol.where(close < close.shift(1), 0)
    df["up_down_volume_ratio"] = up_volume.rolling(10).sum() / (down_volume.rolling(10).sum() + 1e-10)

    # OBV trend
    obv = vol_ind.OnBalanceVolumeIndicator(close, vol)
    df["obv"] = obv.on_balance_volume()
    df["obv_sma"] = df["obv"].rolling(20).mean()
    df["obv_trend"] = (df["obv"] > df["obv_sma"]).astype(int)

    # Candle patterns
    body = abs(close - df["open"])
    full_range = high - low
    df["body_ratio"] = body / (full_range + 1e-10)

    upper_shadow = high - np.maximum(close, df["open"])
    lower_shadow = np.minimum(close, df["open"]) - low
    df["upper_shadow_ratio"] = upper_shadow / (full_range + 1e-10)
    df["lower_shadow_ratio"] = lower_shadow / (full_range + 1e-10)

    df["bullish_candle"] = (close > df["open"]).astype(int)

    df["consecutive_bullish"] = df["bullish_candle"].groupby(
        (df["bullish_candle"] != df["bullish_candle"].shift()).cumsum()
    ).cumsum() * df["bullish_candle"]

    df["consecutive_bearish"] = (1 - df["bullish_candle"]).groupby(
        (df["bullish_candle"] != df["bullish_candle"].shift()).cumsum()
    ).cumsum() * (1 - df["bullish_candle"])

    # Lag features
    for lag in [1, 2, 3]:
        df[f"return_1_lag{lag}"] = df["return_1"].shift(lag)
        df[f"rsi_14_lag{lag}"] = df["rsi_14"].shift(lag)
        df[f"volume_ratio_lag{lag}"] = df["volume_ratio"].shift(lag)

    return df


def generate_reasoning(df: pd.DataFrame, direction: str) -> str:
    """Generate human-readable reasoning"""
    latest = df.iloc[-1]
    reasons = []

    rsi = latest.get("rsi_14", 50)
    if direction == "LONG" and rsi < 40:
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif direction == "SHORT" and rsi > 60:
        reasons.append(f"RSI overbought ({rsi:.0f})")

    macd_diff = latest.get("macd_diff", 0)
    if direction == "LONG" and macd_diff > 0:
        reasons.append("MACD bullish")
    elif direction == "SHORT" and macd_diff < 0:
        reasons.append("MACD bearish")

    ema_ratio = latest.get("ema_ratio_8_21", 0)
    if direction == "LONG" and ema_ratio > 0:
        reasons.append("Short-term uptrend")
    elif direction == "SHORT" and ema_ratio < 0:
        reasons.append("Short-term downtrend")

    bb_pos = latest.get("bb_position_20", 0.5)
    if direction == "LONG" and bb_pos < 0.3:
        reasons.append("Near lower BB")
    elif direction == "SHORT" and bb_pos > 0.7:
        reasons.append("Near upper BB")

    if not reasons:
        reasons.append("Technical confluence")

    return " | ".join(reasons[:3])


def format_signal(signal: dict) -> str:
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
    if timeframe.lower() == "3d":
        tf_label = "[3D SIGNAL]"
        trade_type = "Position Trade"
    elif timeframe.lower() in ["1d", "24h"]:
        tf_label = "[24H SIGNAL]"
        trade_type = "Position Trade"
    elif timeframe.lower() == "1h":
        tf_label = "[1H SIGNAL]"
        trade_type = "Day Trade"
    else:
        tf_label = "[4H SIGNAL]"
        trade_type = "Swing Trade"

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

    if valid_until:
        try:
            dt = datetime.fromisoformat(valid_until.replace('Z', '+00:00'))
            valid_str = dt.strftime("%H:%M UTC")
        except:
            valid_str = valid_until[:16]
    else:
        valid_str = "N/A"

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

_⚠️ Not financial advice. Trade responsibly._
_📈 @QuantumTradingAIX_
"""
    return message.strip()


class SignalScheduler:
    """Scheduler for automated signal generation (1H, 4H, 3D and 24H)"""

    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.redis_client = redis.Redis.from_url(settings.redis_url)
        self.scheduler = AsyncIOScheduler()
        self.session: Optional[aiohttp.ClientSession] = None
        self.models_1h: Dict[str, Dict] = {}  # 1H models for day traders
        self.models_4h: Dict[str, Dict] = {}
        self.models_3d: Dict[str, Dict] = {}  # 3D models for position traders
        self.models_1d: Dict[str, Dict] = {}

    async def start(self):
        """Initialize the scheduler"""
        self.session = aiohttp.ClientSession()

        # Load all models (1H, 4H and 24H)
        self.load_all_models()

        # Schedule 1H signal generation at 5 minutes past every hour
        # 00:05, 01:05, 02:05, ... 23:05 UTC
        self.scheduler.add_job(
            self.generate_and_send_signals_1h,
            CronTrigger(minute=5),  # Every hour at :05
            id="signal_generation_1h",
            name="1H Signal Generation",
        )

        # Schedule 4H signal generation at 5 minutes past each 4-hour candle close
        # 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC
        self.scheduler.add_job(
            self.generate_and_send_signals_4h,
            CronTrigger(hour="0,4,8,12,16,20", minute=5),
            id="signal_generation_4h",
            name="4H Signal Generation",
        )

        # Schedule 24H signal generation at 00:05 UTC (daily candle close)
        self.scheduler.add_job(
            self.generate_and_send_signals_1d,
            CronTrigger(hour=0, minute=5),
            id="signal_generation_1d",
            name="24H Signal Generation",
        )

        # Schedule 3D signal generation at 00:10 UTC (once per day, but signals valid 72h)
        # 3D is for position traders who want wider targets and longer hold times
        self.scheduler.add_job(
            self.generate_and_send_signals_3d,
            CronTrigger(hour=0, minute=10),
            id="signal_generation_3d",
            name="3D Signal Generation",
        )

        # Also run 4H immediately on startup
        await self.generate_and_send_signals_4h()

        self.scheduler.start()
        logger.info("Scheduler started - 1H (every hour) + 4H (every 4 hours) + 24H (daily at 00:05 UTC) + 3D (daily at 00:10 UTC)")

    async def stop(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
        if self.session:
            await self.session.close()

    def load_all_models(self):
        """Load all trained models (1H, 4H and 24H)"""
        model_dir = settings.model_dir

        for symbol in SYMBOLS:
            # Load 1H models (for day traders)
            model_path_1h = self._find_latest_model(symbol, "1h")
            if model_path_1h:
                try:
                    is_v2 = "v2_" in os.path.basename(model_path_1h)
                    self.models_1h[symbol] = {
                        "long": joblib.load(os.path.join(model_path_1h, "long.joblib")),
                        "short": joblib.load(os.path.join(model_path_1h, "short.joblib")),
                        "features": joblib.load(os.path.join(model_path_1h, "features.joblib")),
                        "config": joblib.load(os.path.join(model_path_1h, "config.joblib")),
                        "is_v2": is_v2,
                    }
                    logger.info(f"Loaded 1H {'v2' if is_v2 else 'v1'} model for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load 1H model for {symbol}: {e}")
            else:
                logger.debug(f"No 1H model found for {symbol}")

            # Load 4H models
            model_path_4h = self._find_latest_model(symbol, "4h")
            if model_path_4h:
                try:
                    is_v2 = "v2_" in os.path.basename(model_path_4h)
                    self.models_4h[symbol] = {
                        "long": joblib.load(os.path.join(model_path_4h, "long.joblib")),
                        "short": joblib.load(os.path.join(model_path_4h, "short.joblib")),
                        "features": joblib.load(os.path.join(model_path_4h, "features.joblib")),
                        "config": joblib.load(os.path.join(model_path_4h, "config.joblib")),
                        "is_v2": is_v2,
                    }
                    logger.info(f"Loaded 4H {'v2' if is_v2 else 'v1'} model for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load 4H model for {symbol}: {e}")
            else:
                logger.warning(f"No 4H model found for {symbol}")

            # Load 24H (1d) models
            model_path_1d = self._find_latest_model(symbol, "1d")
            if model_path_1d:
                try:
                    is_v2 = "v2_" in os.path.basename(model_path_1d)
                    self.models_1d[symbol] = {
                        "long": joblib.load(os.path.join(model_path_1d, "long.joblib")),
                        "short": joblib.load(os.path.join(model_path_1d, "short.joblib")),
                        "features": joblib.load(os.path.join(model_path_1d, "features.joblib")),
                        "config": joblib.load(os.path.join(model_path_1d, "config.joblib")),
                        "is_v2": is_v2,
                    }
                    logger.info(f"Loaded 24H {'v2' if is_v2 else 'v1'} model for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load 24H model for {symbol}: {e}")
            else:
                logger.warning(f"No 24H model found for {symbol}")

            # Load 3D models (for position traders)
            model_path_3d = self._find_latest_model(symbol, "3d")
            if model_path_3d:
                try:
                    is_v2 = "v2_" in os.path.basename(model_path_3d)
                    self.models_3d[symbol] = {
                        "long": joblib.load(os.path.join(model_path_3d, "long.joblib")),
                        "short": joblib.load(os.path.join(model_path_3d, "short.joblib")),
                        "features": joblib.load(os.path.join(model_path_3d, "features.joblib")),
                        "config": joblib.load(os.path.join(model_path_3d, "config.joblib")),
                        "is_v2": is_v2,
                    }
                    logger.info(f"Loaded 3D {'v2' if is_v2 else 'v1'} model for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load 3D model for {symbol}: {e}")
            else:
                logger.debug(f"No 3D model found for {symbol}")

        logger.info(f"Loaded {len(self.models_1h)} 1H, {len(self.models_4h)} 4H, {len(self.models_1d)} 24H, {len(self.models_3d)} 3D models")

    def _find_latest_model(self, symbol: str, timeframe: str) -> Optional[str]:
        """Find latest model for symbol - prefers v2 models"""
        model_dir = settings.model_dir
        if not os.path.exists(model_dir):
            return None

        # Prefer v2 models first
        for prefix in [f"v2_{symbol}_{timeframe}_", f"binary_{symbol}_{timeframe}_"]:
            matching = []
            for name in os.listdir(model_dir):
                if name.startswith(prefix):
                    full_path = os.path.join(model_dir, name)
                    if os.path.isdir(full_path):
                        matching.append(full_path)
            if matching:
                return sorted(matching)[-1]

        return None

    def save_signal_to_db(self, signal: Dict):
        """Save signal, features, entry execution, and entry context for RL."""
        try:
            # Insert signal and get ID
            insert_signal = text("""
                INSERT INTO signals (
                    signal_time, symbol, timeframe, direction, confidence,
                    entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
                    risk_reward_ratio, model_version, reasoning, status, valid_until
                ) VALUES (
                    :signal_time, :symbol, :timeframe, :direction, :confidence,
                    :entry_price, :stop_loss, :tp1, :tp2, :tp3,
                    :rr, :model_version, :reasoning, 'active', :valid_until
                )
                RETURNING id, created_at
            """)

            signal_time = datetime.now(timezone.utc)

            with self.engine.begin() as conn:
                result = conn.execute(insert_signal, {
                    "signal_time": signal_time,
                    "symbol": signal["symbol"],
                    "timeframe": signal["timeframe"],
                    "direction": signal["direction"],
                    "confidence": signal["confidence"],
                    "entry_price": signal["entry_price"],
                    "stop_loss": signal["stop_loss"],
                    "tp1": signal["take_profit_1"],
                    "tp2": signal["take_profit_2"],
                    "tp3": signal["take_profit_3"],
                    "rr": signal["risk_reward"],
                    "model_version": "v2",
                    "reasoning": signal.get("reasoning", ""),
                    "valid_until": signal["valid_until"],
                })
                row = result.fetchone()
                signal_id = row[0]
                signal_created_at = row[1]

                # Save features for RL foundation
                features = signal.get("features", {})
                if features:
                    insert_feature = text("""
                        INSERT INTO signal_features (
                            signal_id, signal_created_at, feature_name,
                            feature_value, feature_importance, feature_rank
                        ) VALUES (
                            :signal_id, :created_at, :name,
                            :value, :importance, :rank
                        )
                        ON CONFLICT (signal_id, signal_created_at, feature_name)
                        DO UPDATE SET feature_value = EXCLUDED.feature_value
                    """)

                    for fname, fdata in features.items():
                        conn.execute(insert_feature, {
                            "signal_id": signal_id,
                            "created_at": signal_created_at,
                            "name": fname,
                            "value": fdata.get("value"),
                            "importance": fdata.get("importance"),
                            "rank": fdata.get("rank"),
                        })

                # Record entry execution for RL
                insert_execution = text("""
                    INSERT INTO trade_executions (
                        signal_id, signal_created_at, action, price,
                        quantity_pct, pnl_percent, cumulative_pnl, timestamp
                    ) VALUES (
                        :signal_id, :created_at, 'entry', :price,
                        100.0, 0.0, 0.0, :timestamp
                    )
                """)
                conn.execute(insert_execution, {
                    "signal_id": signal_id,
                    "created_at": signal_created_at,
                    "price": signal["entry_price"],
                    "timestamp": signal_time,
                })

                # Record entry context for RL
                entry_context = signal.get("entry_context", {})
                if entry_context:
                    insert_context = text("""
                        INSERT INTO trade_context (
                            signal_id, signal_created_at, context_type,
                            volatility_regime, trend_direction, volume_profile,
                            atr_value, atr_percent, volume_ratio, rsi_value,
                            price, timestamp
                        ) VALUES (
                            :signal_id, :created_at, 'entry',
                            :volatility_regime, :trend_direction, :volume_profile,
                            :atr_value, :atr_percent, :volume_ratio, :rsi_value,
                            :price, :timestamp
                        )
                    """)
                    conn.execute(insert_context, {
                        "signal_id": signal_id,
                        "created_at": signal_created_at,
                        "volatility_regime": entry_context.get("volatility_regime"),
                        "trend_direction": entry_context.get("trend_direction"),
                        "volume_profile": entry_context.get("volume_profile"),
                        "atr_value": entry_context.get("atr_value"),
                        "atr_percent": entry_context.get("atr_percent"),
                        "volume_ratio": entry_context.get("volume_ratio"),
                        "rsi_value": entry_context.get("rsi_value"),
                        "price": signal["entry_price"],
                        "timestamp": signal_time,
                    })

            logger.debug(f"Saved signal {signal_id} with {len(features)} features for {signal['symbol']}")
        except Exception as e:
            logger.error(f"Failed to save signal to DB: {e}")

    async def send_telegram_message(self, text: str) -> bool:
        """Send message to Telegram channel"""
        url = f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage"

        try:
            async with self.session.post(url, json={
                "chat_id": settings.telegram_channel_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }) as resp:
                data = await resp.json()
                if data.get("ok"):
                    return True
                else:
                    logger.error(f"Telegram error: {data}")
        except Exception as e:
            logger.error(f"Failed to send to Telegram: {e}")
        return False

    async def generate_and_send_signals_1h(self):
        """Generate 1H signals for day traders and send to Telegram"""
        await self._generate_and_send_signals(
            models=self.models_1h,
            timeframe="1h",
            offset="1h",
            valid_hours=1,
            signal_type="1H"
        )

    async def generate_and_send_signals_4h(self):
        """Generate 4H signals for all symbols and send to Telegram"""
        await self._generate_and_send_signals(
            models=self.models_4h,
            timeframe="4h",
            offset="4h",
            valid_hours=4,
            signal_type="4H"
        )

    async def generate_and_send_signals_1d(self):
        """Generate 24H (daily) signals for all symbols and send to Telegram"""
        await self._generate_and_send_signals(
            models=self.models_1d,
            timeframe="1d",
            offset="1D",
            valid_hours=24,
            signal_type="24H"
        )

    async def generate_and_send_signals_3d(self):
        """Generate 3D signals for position traders and send to Telegram"""
        await self._generate_and_send_signals(
            models=self.models_3d,
            timeframe="3d",
            offset="3D",
            valid_hours=72,  # Valid for 3 days
            signal_type="3D"
        )

    async def _generate_and_send_signals(
        self,
        models: Dict[str, Dict],
        timeframe: str,
        offset: str,
        valid_hours: int,
        signal_type: str
    ):
        """Generate signals for all symbols and send to Telegram"""
        now = datetime.now(timezone.utc)
        logger.info(f"Starting {signal_type} signal generation at {now.isoformat()}")

        signals = []

        for symbol in SYMBOLS:
            if symbol not in models:
                continue

            try:
                signal = self._generate_signal(symbol, timeframe, offset, models, valid_hours)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating {signal_type} signal for {symbol}: {e}")

        logger.info(f"Generated {len(signals)} {signal_type} signals")

        if signals:
            # Save each signal to database and publish to Redis
            # (Telegram bot will handle sending via Redis subscription)
            for signal in signals:
                # Save to database for outcome tracking
                self.save_signal_to_db(signal)

                # Publish to Redis - bot will send to Telegram
                self.redis_client.publish(
                    "quantum:signals",
                    json.dumps(signal)
                )
                await asyncio.sleep(0.5)  # Rate limiting between publishes
        else:
            # Send status update if no signals (scheduler sends this directly)
            status = f"""
📊 *Quantum Trading AI - {signal_type} Update*

🔍 Scanned {len(SYMBOLS)} crypto pairs
⏱ Timeframe: {timeframe.upper()}
📉 No strong signals at {now.strftime('%H:%M UTC')}

_Market conditions not favorable. Stay tuned!_
_📈 @QuantumTradingAIX_
"""
            await self.send_telegram_message(status.strip())

    def _generate_signal(
        self, symbol: str, timeframe: str, offset: str, models: Dict[str, Dict], valid_hours: int
    ) -> Optional[Dict]:
        """Generate signal for a single symbol"""
        model_data = models.get(symbol)
        if not model_data:
            return None

        # Load data
        query = text("""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = :symbol AND timeframe = '1m'
            ORDER BY time DESC
            LIMIT 100000
        """)

        df = pd.read_sql(query, self.engine, params={"symbol": symbol}, parse_dates=["time"])
        df = df.sort_values("time").set_index("time")

        # Resample
        df = df.resample(offset).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

        if len(df) < 100:
            return None

        # Compute features based on model version
        is_v2 = model_data.get("is_v2", False)
        if is_v2:
            df = compute_features_v2(df)
        else:
            df = compute_features(df)

        # Get latest features
        feature_cols = model_data["features"]
        latest_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
        latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions
        long_proba = model_data["long"].predict_proba(latest_features)[0, 1]
        short_proba = model_data["short"].predict_proba(latest_features)[0, 1]

        config = model_data["config"]
        long_threshold = config.get("long_threshold", 0.5)
        short_threshold = config.get("short_threshold", 0.5)

        # Determine direction
        direction = None
        confidence = 0

        if long_proba >= long_threshold and long_proba > short_proba:
            direction = "LONG"
            confidence = long_proba * 100
        elif short_proba >= short_threshold:
            direction = "SHORT"
            confidence = short_proba * 100

        if not direction:
            return None

        # Calculate levels
        latest = df.iloc[-1]
        price = latest["close"]
        atr = latest["atr_14"]

        if direction == "LONG":
            sl = price - atr * ATR_SL_MULTIPLIER
            tp1 = price + atr * ATR_TP1_MULTIPLIER
            tp2 = price + atr * ATR_TP2_MULTIPLIER
            tp3 = price + atr * ATR_TP3_MULTIPLIER
        else:
            sl = price + atr * ATR_SL_MULTIPLIER
            tp1 = price - atr * ATR_TP1_MULTIPLIER
            tp2 = price - atr * ATR_TP2_MULTIPLIER
            tp3 = price - atr * ATR_TP3_MULTIPLIER

        risk = abs(price - sl)
        reward = abs(tp2 - price)
        rr = reward / risk if risk > 0 else 0

        now = datetime.now(timezone.utc)
        valid_until = now + timedelta(hours=valid_hours)

        # Extract feature values and importances for RL foundation
        feature_data = {}
        for i, fname in enumerate(feature_cols):
            feature_data[fname] = {
                "value": float(latest_features[0, i]),
                "importance": None,  # Will be added if model supports it
                "rank": None,
            }

        # Try to get feature importances from model
        try:
            long_model = model_data["long"]
            if hasattr(long_model, "feature_importances_"):
                importances = long_model.feature_importances_
                # Create ranking (1 = most important)
                ranks = len(importances) - np.argsort(np.argsort(importances))
                for i, fname in enumerate(feature_cols):
                    feature_data[fname]["importance"] = float(importances[i])
                    feature_data[fname]["rank"] = int(ranks[i])
        except Exception as e:
            logger.debug(f"Could not extract feature importances: {e}")

        # Calculate entry context for RL training
        entry_context = self._calculate_market_context(latest, price)

        return {
            "id": str(uuid4()),
            "symbol": symbol,
            "direction": direction,
            "timeframe": timeframe,
            "entry_price": float(price),
            "stop_loss": float(sl),
            "take_profit_1": float(tp1),
            "take_profit_2": float(tp2),
            "take_profit_3": float(tp3),
            "confidence": float(confidence),
            "risk_reward": float(rr),
            "reasoning": generate_reasoning(df, direction),
            "created_at": now.isoformat(),
            "valid_until": valid_until.isoformat(),
            "features": feature_data,  # Include features for storage
            "entry_context": entry_context,  # Include context for RL
        }

    def _calculate_market_context(self, latest: pd.Series, price: float) -> Dict[str, Any]:
        """Calculate market context (volatility regime, trend, volume) for RL."""
        try:
            # ATR percent for volatility regime
            atr_value = latest.get("atr_14", 0)
            atr_percent = (atr_value / price * 100) if price > 0 else 0

            # Volatility regime: low (<1%), medium (1-3%), high (>3%)
            if atr_percent < 1.0:
                volatility_regime = "low"
            elif atr_percent < 3.0:
                volatility_regime = "medium"
            else:
                volatility_regime = "high"

            # Trend direction based on EMA position
            price_vs_ema = latest.get("price_vs_ema_21", latest.get("price_vs_ema_50", 0))
            if price_vs_ema > 0.01:  # >1% above EMA
                trend_direction = "bullish"
            elif price_vs_ema < -0.01:  # >1% below EMA
                trend_direction = "bearish"
            else:
                trend_direction = "sideways"

            # Volume profile based on volume ratio
            volume_ratio = latest.get("volume_sma_ratio_20", latest.get("volume_ratio", 1.0))
            if volume_ratio < 0.7:
                volume_profile = "low"
            elif volume_ratio > 1.5:
                volume_profile = "high"
            else:
                volume_profile = "normal"

            # RSI value
            rsi_value = latest.get("rsi_14", 50)

            return {
                "volatility_regime": volatility_regime,
                "trend_direction": trend_direction,
                "volume_profile": volume_profile,
                "atr_value": float(atr_value) if atr_value else None,
                "atr_percent": float(atr_percent) if atr_percent else None,
                "volume_ratio": float(volume_ratio) if volume_ratio else None,
                "rsi_value": float(rsi_value) if rsi_value else None,
            }
        except Exception as e:
            logger.warning(f"Failed to calculate market context: {e}")
            return {}

    async def run(self):
        """Main run loop"""
        await self.start()

        try:
            # Keep running
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            logger.info("Scheduler shutting down...")
        finally:
            await self.stop()
