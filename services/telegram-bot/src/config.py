"""
Quantum Trading AI - Telegram Bot Configuration
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


# Top 50 symbols available for trading (by market cap)
AVAILABLE_SYMBOLS = [
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

# Available timeframes
AVAILABLE_TIMEFRAMES = ["1h", "4h", "1d", "3d"]


class Settings(BaseSettings):
    """Configuration for Telegram bot service"""

    # Telegram (REQUIRED - must be set via environment variable)
    telegram_token: str = Field(
        ...,  # Required field - no default
        alias="TELEGRAM_BOT_TOKEN"
    )
    telegram_channel_id: str = Field(
        ...,  # Required field - no default
        alias="TELEGRAM_CHANNEL_ID"
    )

    # Redis
    redis_host: str = Field(default="redis", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")

    # Database
    database_url: str = Field(
        ...,  # Required - must be set via environment variable
        alias="DATABASE_URL"
    )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
