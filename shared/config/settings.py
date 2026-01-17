"""
Quantum Trading AI - Core Configuration
========================================
Central configuration for all services.
"""
import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# ASSET CONFIGURATION
# =============================================================================

class AssetClass(Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX = "forex"
    COMMODITIES = "commodities"
    INDICES = "indices"


# MVP: Top 20 Crypto Pairs (Binance USDT pairs)
CRYPTO_PAIRS_MVP: List[str] = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "SOLUSDT",   # Solana
    "BNBUSDT",   # Binance Coin
    "XRPUSDT",   # Ripple
    "ADAUSDT",   # Cardano
    "DOGEUSDT",  # Dogecoin
    "MATICUSDT", # Polygon (now POL but still MATIC on Binance)
    "DOTUSDT",   # Polkadot
    "AVAXUSDT",  # Avalanche
    "LINKUSDT",  # Chainlink
    "UNIUSDT",   # Uniswap
    "ATOMUSDT",  # Cosmos
    "LTCUSDT",   # Litecoin
    "NEARUSDT",  # Near Protocol
    "FTMUSDT",   # Fantom
    "ALGOUSDT",  # Algorand
    "AAVEUSDT",  # Aave
    "SANDUSDT",  # The Sandbox
    "MANAUSDT",  # Decentraland
]

# Free tier: Top 10 coins
FREE_TIER_PAIRS: List[str] = CRYPTO_PAIRS_MVP[:10]

# Full list for Phase 2 expansion
CRYPTO_PAIRS_EXTENDED: List[str] = CRYPTO_PAIRS_MVP + [
    "SHIBUSDT", "TRXUSDT", "XLMUSDT", "VETUSDT", "ICPUSDT",
    "FILUSDT", "HBARUSDT", "EGLDUSDT", "THETAUSDT", "AXSUSDT",
    "XTZUSDT", "EOSUSDT", "ARUSDT", "GRTUSDT", "FLOWUSDT",
    "ENJUSDT", "CHZUSDT", "BATUSDT", "ZILUSDT", "ZECUSDT",
    "DASHUSDT", "WAVESUSDT", "NEOUSDT", "IOSTUSDT", "ONTUSDT",
    "QTUMUSDT", "KSMUSDT", "RUNEUSDT", "CRVUSDT", "COMPUSDT",
]


# =============================================================================
# TIMEFRAME CONFIGURATION
# =============================================================================

class Timeframe(Enum):
    """Supported trading timeframes"""
    M1 = "1m"    # 1 minute (data collection only)
    M5 = "5m"    # 5 minutes
    M15 = "15m"  # 15 minutes
    H1 = "1h"    # 1 hour
    H4 = "4h"    # 4 hours (MVP)
    D1 = "1d"    # 1 day / 24 hours (MVP)
    W1 = "1w"    # 1 week


# MVP timeframes for signal generation
SIGNAL_TIMEFRAMES: List[str] = ["4h", "1d"]

# Timeframe to minutes mapping
TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


# =============================================================================
# SIGNAL CONFIGURATION
# =============================================================================

@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    # Confidence threshold (0-100)
    confidence_threshold: float = 70.0

    # Minimum Risk:Reward ratio
    min_risk_reward: float = 2.0

    # ATR multipliers for SL/TP
    atr_sl_multiplier: float = 1.5
    atr_tp1_multiplier: float = 2.0
    atr_tp2_multiplier: float = 3.0
    atr_tp3_multiplier: float = 4.0

    # Maximum signals per day
    max_signals_per_day: int = 50

    # Minimum volume filter (relative to 20-day average)
    min_volume_ratio: float = 0.5

    # Maximum position size recommendation
    max_position_size_pct: float = 5.0

    # Signal validity duration (hours)
    signal_validity_hours: Dict[str, int] = field(default_factory=lambda: {
        "4h": 4,
        "1d": 24,
    })


# =============================================================================
# TIER CONFIGURATION
# =============================================================================

@dataclass
class TierConfig:
    """Configuration for subscription tiers"""
    name: str
    price_usd: float
    pairs: List[str]
    timeframes: List[str]
    max_signals_per_day: int
    delay_minutes: int
    api_access: bool
    priority_delivery: bool
    features: List[str]


TIER_CONFIGS: Dict[str, TierConfig] = {
    "free": TierConfig(
        name="Free",
        price_usd=0,
        pairs=FREE_TIER_PAIRS,
        timeframes=["4h", "1d"],
        max_signals_per_day=10,
        delay_minutes=15,
        api_access=False,
        priority_delivery=False,
        features=["basic_signals", "telegram"],
    ),
    "starter": TierConfig(
        name="Starter",
        price_usd=19,
        pairs=CRYPTO_PAIRS_MVP,
        timeframes=["4h", "1d"],
        max_signals_per_day=30,
        delay_minutes=0,
        api_access=False,
        priority_delivery=False,
        features=["basic_signals", "telegram", "email", "detailed_analysis"],
    ),
    "pro": TierConfig(
        name="Pro",
        price_usd=49,
        pairs=CRYPTO_PAIRS_EXTENDED,
        timeframes=["1h", "4h", "1d"],
        max_signals_per_day=100,
        delay_minutes=0,
        api_access=True,
        priority_delivery=False,
        features=["all_signals", "telegram", "email", "discord", "api", "backtests", "alerts"],
    ),
    "elite": TierConfig(
        name="Elite",
        price_usd=149,
        pairs=CRYPTO_PAIRS_EXTENDED,
        timeframes=["1h", "4h", "1d", "1w"],
        max_signals_per_day=999,
        delay_minutes=0,
        api_access=True,
        priority_delivery=True,
        features=["all_signals", "all_channels", "api", "backtests", "alerts", "support", "community"],
    ),
}


# =============================================================================
# ML CONFIGURATION
# =============================================================================

@dataclass
class MLConfig:
    """Configuration for machine learning pipeline"""
    # Target definition
    target_threshold_pct: float = 2.0  # +/-2% for LONG/SHORT classification

    # Training window
    training_days: int = 180
    validation_days: int = 30
    purge_days: int = 1  # Gap between train/val to prevent leakage

    # Walk-forward CV
    n_cv_splits: int = 5

    # Model hyperparameters (LightGBM)
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    })

    # Model hyperparameters (XGBoost)
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    })

    # Retraining schedule (hours)
    retrain_interval_hours: Dict[str, int] = field(default_factory=lambda: {
        "4h": 24 * 7,   # Weekly
        "1d": 24 * 14,  # Bi-weekly
    })


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Technical indicators to compute
TECHNICAL_INDICATORS: Dict[str, Dict[str, Any]] = {
    # Momentum
    "RSI": {"periods": [7, 14, 21]},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "ROC": {"periods": [5, 10, 20]},
    "MFI": {"period": 14},
    "WILLR": {"period": 14},
    "CCI": {"period": 20},

    # Trend
    "ADX": {"period": 14},
    "SUPERTREND": {"period": 10, "multiplier": 3},
    "EMA": {"periods": [8, 21, 55, 200]},
    "SMA": {"periods": [20, 50, 200]},

    # Volatility
    "ATR": {"periods": [7, 14, 21]},
    "BBANDS": {"period": 20, "std": 2},

    # Volume
    "OBV": {},
    "VWAP": {},
    "VOLUME_SMA": {"periods": [20, 50]},
}

# Feature groups
FEATURE_GROUPS: List[str] = [
    "returns",
    "volatility",
    "momentum",
    "trend",
    "volume",
    "market_structure",
    "cross_asset",
]


# =============================================================================
# BINANCE API CONFIGURATION
# =============================================================================

BINANCE_CONFIG = {
    "base_url": "https://api.binance.com",
    "ws_url": "wss://stream.binance.com:9443/ws",
    "rest_rate_limit": 1200,  # requests per minute
    "ws_connections_limit": 5,
    "kline_limit": 1000,  # max candles per request
}


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

def get_db_config() -> Dict[str, Any]:
    """Get database configuration from environment - REQUIRED vars"""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "database": os.getenv("DB_NAME", "quantum_trading"),
    }


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration from environment"""
    return {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": int(os.getenv("REDIS_DB", "0")),
    }


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "/app/logs/quantum.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "handlers": ["console", "file"],
    },
}


# =============================================================================
# SIGNAL FORMATTING
# =============================================================================

SIGNAL_EMOJI = {
    "LONG": "🟢",
    "SHORT": "🔴",
    "NEUTRAL": "🟡",
    "high_confidence": "🔥",
    "new": "🆕",
    "updated": "🔄",
    "closed": "✅",
    "stopped": "❌",
}

SIGNAL_TEMPLATE = """
{emoji} *{direction} Signal* | {symbol}

📊 *Timeframe:* {timeframe}
💰 *Entry:* ${entry_price:.4f}
🎯 *Take Profits:*
   TP1: ${tp1:.4f} (+{tp1_pct:.1f}%)
   TP2: ${tp2:.4f} (+{tp2_pct:.1f}%)
   TP3: ${tp3:.4f} (+{tp3_pct:.1f}%)
🛑 *Stop Loss:* ${stop_loss:.4f} (-{sl_pct:.1f}%)
📏 *R:R Ratio:* {risk_reward:.1f}:1
🎯 *Confidence:* {confidence:.0f}%
⏰ *Valid Until:* {valid_until}

💡 *Reasoning:* {reasoning}

_⚠️ Not financial advice. Trade responsibly._
"""
