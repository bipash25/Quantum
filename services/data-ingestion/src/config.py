"""
Quantum Trading AI - Data Ingestion Configuration
==================================================
Settings for the data ingestion service.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """Data ingestion service settings"""

    # Database (REQUIRED - must be set via environment variables)
    db_host: str = Field(..., alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_user: str = Field(..., alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")
    db_name: str = Field(..., alias="DB_NAME")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")

    # Binance API
    binance_rest_url: str = "https://api.binance.com"
    binance_ws_url: str = "wss://stream.binance.com:9443"

    # Symbols to track (MVP: top 20)
    symbols: List[str] = Field(default=[
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
        "FTMUSDT", "ALGOUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT",
    ])

    # Collection settings
    candle_interval: str = "1m"  # Collect 1-minute candles
    buffer_flush_interval: int = 60  # Flush to DB every 60 seconds

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def db_dsn(self) -> str:
        """Database connection string"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
