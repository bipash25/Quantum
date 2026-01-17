"""
Quantum Trading AI - Telegram Bot Configuration
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


# Available symbols for trading
AVAILABLE_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT",
    "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "LTCUSDT",
    "UNIUSDT", "NEARUSDT", "ALGOUSDT", "AAVEUSDT"
]

# Available timeframes
AVAILABLE_TIMEFRAMES = ["4h", "1d"]


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
