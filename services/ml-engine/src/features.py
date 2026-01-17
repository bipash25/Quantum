"""
Quantum Trading AI - Feature Engineering
=========================================
Computes technical indicators and features for ML models.

Uses the `ta` library for technical analysis calculations.
All features are computed with NO LOOK-AHEAD BIAS.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ta import momentum, trend, volatility, volume

from .config import settings


logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature computation"""
    version: str = "v1.0.0"
    return_periods: List[int] = field(default_factory=lambda: [5, 15, 60, 240, 1440])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    ema_periods: List[int] = field(default_factory=lambda: [8, 21, 55, 200])
    atr_periods: List[int] = field(default_factory=lambda: [7, 14, 21])


class FeatureEngine:
    """
    Computes all features for trading signals.

    IMPORTANT: All features are computed using ONLY past data.
    """

    FEATURE_NAMES = [
        # Returns
        "return_5", "return_15", "return_60", "return_240", "return_1440",
        "log_return_60", "log_return_240",
        # Volatility
        "atr_7", "atr_14", "atr_21", "atr_pct_14",
        "volatility_60", "volatility_240",
        "bb_width_20", "bb_position_20",
        # Momentum
        "rsi_7", "rsi_14", "rsi_21",
        "macd", "macd_signal", "macd_diff",
        "roc_5", "roc_10", "roc_20",
        "stoch_k", "stoch_d",
        "williams_r",
        # Trend
        "adx_14",
        "ema_ratio_8_21", "ema_ratio_21_55",
        "price_vs_ema_21", "price_vs_ema_55",
        "trend_strength",
        # Volume
        "volume_sma_ratio_20", "volume_momentum",
        "obv_slope",
        # Market structure
        "distance_to_high_20", "distance_to_low_20",
        "range_position",
    ]

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names = self.FEATURE_NAMES
        logger.info(f"FeatureEngine initialized with {len(self.feature_names)} features")

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for the given OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with all computed features.
        """
        if len(df) < 200:
            raise ValueError(f"Need at least 200 candles, got {len(df)}")

        df = df.copy()
        high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]

        # === RETURNS ===
        for period in self.config.return_periods:
            df[f"return_{period}"] = close.pct_change(period)

        df["log_return_60"] = np.log(close / close.shift(60))
        df["log_return_240"] = np.log(close / close.shift(240))

        # === VOLATILITY ===
        for period in self.config.atr_periods:
            atr_indicator = volatility.AverageTrueRange(high, low, close, window=period)
            df[f"atr_{period}"] = atr_indicator.average_true_range()

        df["atr_pct_14"] = df["atr_14"] / close * 100

        returns = close.pct_change()
        df["volatility_60"] = returns.rolling(60).std() * np.sqrt(60)
        df["volatility_240"] = returns.rolling(240).std() * np.sqrt(240)

        # Bollinger Bands
        bb = volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_width_20"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        bb_range = bb.bollinger_hband() - bb.bollinger_lband()
        df["bb_position_20"] = (close - bb.bollinger_lband()) / (bb_range + 1e-10)

        # === MOMENTUM ===
        for period in self.config.rsi_periods:
            rsi = momentum.RSIIndicator(close, window=period)
            df[f"rsi_{period}"] = rsi.rsi()

        # MACD
        macd = trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # Rate of Change
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = momentum.ROCIndicator(close, window=period).roc()

        # Stochastic
        stoch = momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Williams %R
        df["williams_r"] = momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

        # === TREND ===
        adx = trend.ADXIndicator(high, low, close, window=14)
        df["adx_14"] = adx.adx()

        # EMAs
        emas = {}
        for period in self.config.ema_periods:
            ema = trend.EMAIndicator(close, window=period)
            emas[period] = ema.ema_indicator()

        df["ema_ratio_8_21"] = emas[8] / emas[21] - 1
        df["ema_ratio_21_55"] = emas[21] / emas[55] - 1
        df["price_vs_ema_21"] = close / emas[21] - 1
        df["price_vs_ema_55"] = close / emas[55] - 1

        # Trend strength (ADX-based)
        plus_di = adx.adx_pos()
        minus_di = adx.adx_neg()
        df["trend_strength"] = df["adx_14"] * np.sign(plus_di - minus_di) / 100

        # === VOLUME ===
        vol_sma = vol.rolling(20).mean()
        df["volume_sma_ratio_20"] = vol / (vol_sma + 1e-10)
        df["volume_momentum"] = vol.pct_change(5)

        # OBV slope
        obv = volume.OnBalanceVolumeIndicator(close, vol)
        obv_values = obv.on_balance_volume()
        df["obv_slope"] = obv_values.diff(10) / (obv_values.rolling(10).mean().abs() + 1e-10)

        # === MARKET STRUCTURE ===
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()

        df["distance_to_high_20"] = (high_20 - close) / close
        df["distance_to_low_20"] = (close - low_20) / close
        df["range_position"] = (close - low_20) / (high_20 - low_20 + 1e-10)

        return df

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        dropna: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """Get feature matrix ready for ML training."""
        feature_cols = [col for col in self.feature_names if col in df.columns]
        X = df[feature_cols].copy()

        # Handle infinities and NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        if dropna:
            X = X.fillna(0)

        return X.values, feature_cols

    def compute_target(
        self,
        df: pd.DataFrame,
        timeframe_minutes: int = 1,
        threshold_pct: float = 2.0
    ) -> pd.Series:
        """
        Compute classification target for ML training.

        Target classes:
        - 0: SHORT (price drops > threshold%)
        - 1: NEUTRAL (price stays within threshold%)
        - 2: LONG (price rises > threshold%)
        """
        close = df["close"]

        # Calculate forward return
        forward_return = close.shift(-timeframe_minutes) / close - 1
        forward_return_pct = forward_return * 100

        # Classify
        target = pd.Series(1, index=df.index)  # Default: NEUTRAL
        target[forward_return_pct > threshold_pct] = 2   # LONG
        target[forward_return_pct < -threshold_pct] = 0  # SHORT

        # Mark last rows as NaN (no future data)
        target.iloc[-timeframe_minutes:] = np.nan

        return target
