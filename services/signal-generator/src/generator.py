"""
Quantum Trading AI - Signal Generator
======================================
Generates trading signals from ML model predictions.

Process:
1. Load latest candle data from database
2. Compute technical features
3. Run predictions through trained models
4. Apply confidence thresholds and filters
5. Calculate entry/SL/TP levels
6. Publish signals to Redis
"""
import asyncio
import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from uuid import uuid4

import numpy as np
import pandas as pd
import redis
import joblib
from sqlalchemy import create_engine, text
from ta import momentum, trend, volatility

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


# ATR multipliers for SL/TP
ATR_SL_MULTIPLIER = 1.5
ATR_TP1_MULTIPLIER = 2.0
ATR_TP2_MULTIPLIER = 3.0
ATR_TP3_MULTIPLIER = 4.5


@dataclass
class Signal:
    """Trading signal data"""
    id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    confidence: float
    atr: float
    risk_reward: float
    reasoning: str
    created_at: str
    valid_until: str

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelLoader:
    """Loads and manages trained ML models"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models: Dict[str, Dict] = {}

    def load_model(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Load model for a specific symbol and timeframe"""
        key = f"{symbol}_{timeframe}"

        if key in self.models:
            return self.models[key]

        # Find latest model for this symbol/timeframe
        model_path = self._find_latest_model(symbol, timeframe)

        if not model_path:
            logger.warning(f"No model found for {symbol} {timeframe}")
            return None

        try:
            model_data = {
                "long": joblib.load(os.path.join(model_path, "long.joblib")),
                "short": joblib.load(os.path.join(model_path, "short.joblib")),
                "features": joblib.load(os.path.join(model_path, "features.joblib")),
                "config": joblib.load(os.path.join(model_path, "config.joblib")),
            }
            self.models[key] = model_data
            logger.info(f"Loaded model for {symbol} {timeframe}")
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None

    def _find_latest_model(self, symbol: str, timeframe: str) -> Optional[str]:
        """Find the latest trained model for a symbol"""
        if not os.path.exists(self.model_dir):
            return None

        pattern = f"binary_{symbol}_{timeframe}_"
        matching = []

        for name in os.listdir(self.model_dir):
            if name.startswith(pattern):
                full_path = os.path.join(self.model_dir, name)
                if os.path.isdir(full_path):
                    matching.append(full_path)

        if not matching:
            return None

        # Return the latest (sorted by name which includes timestamp)
        return sorted(matching)[-1]


class SignalGenerator:
    """Generates trading signals from ML predictions"""

    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.redis_client = redis.Redis.from_url(settings.redis_url)
        self.model_loader = ModelLoader(settings.model_dir)

        # Symbols to generate signals for
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT",
            "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
            "ALGOUSDT", "AAVEUSDT",
        ]

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical features from candle data"""
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
        macd_ind = trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_diff"] = macd_ind.macd_diff()

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

    def load_candle_data(self, symbol: str, timeframe: str, bars: int = 300) -> pd.DataFrame:
        """Load recent candle data from database"""
        offset = {"1h": "1h", "4h": "4h", "1d": "1D"}.get(timeframe, "4h")

        # Load 1-minute data and resample
        query = text("""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = :symbol AND timeframe = '1m'
            ORDER BY time DESC
            LIMIT :limit
        """)

        df = pd.read_sql(query, self.engine, params={
            "symbol": symbol, "limit": bars * 240 + 500
        }, parse_dates=["time"])

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("time")
        df.set_index("time", inplace=True)

        # Resample
        resampled = df.resample(offset).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

        return resampled

    def generate_reasoning(self, df: pd.DataFrame, direction: str) -> str:
        """Generate human-readable reasoning for the signal"""
        latest = df.iloc[-1]
        reasons = []

        # RSI
        rsi = latest.get("rsi_14", 50)
        if direction == "LONG" and rsi < 40:
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif direction == "SHORT" and rsi > 60:
            reasons.append(f"RSI overbought ({rsi:.0f})")

        # MACD
        macd_diff = latest.get("macd_diff", 0)
        if direction == "LONG" and macd_diff > 0:
            reasons.append("MACD bullish crossover")
        elif direction == "SHORT" and macd_diff < 0:
            reasons.append("MACD bearish crossover")

        # Trend
        ema_ratio = latest.get("ema_ratio_8_21", 0)
        if direction == "LONG" and ema_ratio > 0:
            reasons.append("Short-term uptrend")
        elif direction == "SHORT" and ema_ratio < 0:
            reasons.append("Short-term downtrend")

        # BB position
        bb_pos = latest.get("bb_position_20", 0.5)
        if direction == "LONG" and bb_pos < 0.3:
            reasons.append("Near lower Bollinger Band")
        elif direction == "SHORT" and bb_pos > 0.7:
            reasons.append("Near upper Bollinger Band")

        if not reasons:
            reasons.append("Technical confluence detected")

        return " | ".join(reasons[:3])

    def calculate_levels(self, price: float, atr: float, direction: str) -> Dict[str, float]:
        """Calculate stop loss and take profit levels"""
        if direction == "LONG":
            return {
                "stop_loss": price - atr * ATR_SL_MULTIPLIER,
                "take_profit_1": price + atr * ATR_TP1_MULTIPLIER,
                "take_profit_2": price + atr * ATR_TP2_MULTIPLIER,
                "take_profit_3": price + atr * ATR_TP3_MULTIPLIER,
            }
        else:  # SHORT
            return {
                "stop_loss": price + atr * ATR_SL_MULTIPLIER,
                "take_profit_1": price - atr * ATR_TP1_MULTIPLIER,
                "take_profit_2": price - atr * ATR_TP2_MULTIPLIER,
                "take_profit_3": price - atr * ATR_TP3_MULTIPLIER,
            }

    def generate_signal_for_symbol(self, symbol: str, timeframe: str) -> Optional[Signal]:
        """Generate signal for a single symbol"""
        # Load model
        model_data = self.model_loader.load_model(symbol, timeframe)
        if not model_data:
            return None

        # Load candle data
        df = self.load_candle_data(symbol, timeframe)
        if len(df) < 100:
            logger.warning(f"Not enough data for {symbol}")
            return None

        # Compute features
        df = self.compute_features(df)

        # Get latest features
        feature_cols = model_data["features"]
        latest_features = df[feature_cols].iloc[-1].values.reshape(1, -1)

        # Handle NaN
        latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions
        long_model = model_data["long"]
        short_model = model_data["short"]
        config = model_data["config"]

        long_proba = long_model.predict_proba(latest_features)[0, 1]
        short_proba = short_model.predict_proba(latest_features)[0, 1]

        # Get thresholds from config
        long_threshold = config.get("long_threshold", 0.55)
        short_threshold = config.get("short_threshold", 0.55)

        # Determine direction
        direction = None
        confidence = 0

        if long_proba >= long_threshold and long_proba > short_proba:
            direction = "LONG"
            confidence = long_proba
        elif short_proba >= short_threshold and short_proba > long_proba:
            direction = "SHORT"
            confidence = short_proba
        else:
            # No signal
            return None

        # Volume filter
        vol_ratio = df["volume_sma_ratio_20"].iloc[-1]
        if vol_ratio < settings.min_volume_ratio:
            logger.info(f"{symbol}: Low volume ({vol_ratio:.2f}), skipping")
            return None

        # Calculate levels
        latest = df.iloc[-1]
        price = latest["close"]
        atr = latest["atr_14"]

        levels = self.calculate_levels(price, atr, direction)

        # Calculate risk/reward
        risk = abs(price - levels["stop_loss"])
        reward = abs(levels["take_profit_2"] - price)
        risk_reward = reward / risk if risk > 0 else 0

        # Generate reasoning
        reasoning = self.generate_reasoning(df, direction)

        # Validity period
        now = datetime.now(timezone.utc)
        validity_hours = 4 if timeframe == "4h" else 24
        valid_until = now + timedelta(hours=validity_hours)

        return Signal(
            id=str(uuid4()),
            symbol=symbol,
            direction=direction,
            timeframe=timeframe,
            entry_price=price,
            stop_loss=levels["stop_loss"],
            take_profit_1=levels["take_profit_1"],
            take_profit_2=levels["take_profit_2"],
            take_profit_3=levels["take_profit_3"],
            confidence=confidence * 100,  # As percentage
            atr=atr,
            risk_reward=risk_reward,
            reasoning=reasoning,
            created_at=now.isoformat(),
            valid_until=valid_until.isoformat(),
        )

    def publish_signal(self, signal: Signal):
        """Publish signal to Redis for the Telegram bot"""
        channel = "quantum:signals"
        message = json.dumps(signal.to_dict())
        self.redis_client.publish(channel, message)

        # Also store in a list for persistence
        self.redis_client.lpush("quantum:signals:list", message)
        self.redis_client.ltrim("quantum:signals:list", 0, 999)  # Keep last 1000

        logger.info(f"Published signal: {signal.symbol} {signal.direction} (conf: {signal.confidence:.1f}%)")

    def save_signal_to_db(self, signal: Signal):
        """Save signal to database for tracking"""
        query = text("""
            INSERT INTO signals (
                id, symbol, direction, timeframe, entry_price, stop_loss,
                take_profit_1, take_profit_2, take_profit_3, confidence,
                atr, risk_reward, reasoning, created_at, valid_until
            ) VALUES (
                :id, :symbol, :direction, :timeframe, :entry_price, :stop_loss,
                :take_profit_1, :take_profit_2, :take_profit_3, :confidence,
                :atr, :risk_reward, :reasoning, :created_at, :valid_until
            )
        """)

        with self.engine.connect() as conn:
            conn.execute(query, signal.to_dict())
            conn.commit()

    def run_once(self) -> List[Signal]:
        """Generate signals for all symbols once"""
        signals = []
        timeframe = settings.signal_timeframe

        logger.info(f"Generating signals for {len(self.symbols)} symbols ({timeframe})...")

        for symbol in self.symbols:
            try:
                signal = self.generate_signal_for_symbol(symbol, timeframe)
                if signal:
                    signals.append(signal)
                    self.publish_signal(signal)
                    try:
                        self.save_signal_to_db(signal)
                    except Exception as e:
                        logger.warning(f"Failed to save signal to DB: {e}")
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        logger.info(f"Generated {len(signals)} signals")
        return signals

    async def run_loop(self):
        """Run signal generation in a loop"""
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")

            await asyncio.sleep(settings.generation_interval)
