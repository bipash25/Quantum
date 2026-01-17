"""
Quantum Trading AI - ML Engine Configuration
=============================================
Settings for the ML engine service.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Dict, Any
import os


class Settings(BaseSettings):
    """ML engine service settings"""

    # Database (REQUIRED - must be set via environment variables)
    db_host: str = Field(..., alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_user: str = Field(..., alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")
    db_name: str = Field(..., alias="DB_NAME")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        alias="CELERY_RESULT_BACKEND"
    )

    # Model settings
    model_dir: str = "/app/models"

    # Symbols
    symbols: List[str] = Field(default=[
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
        "FTMUSDT", "ALGOUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT",
    ])

    # Training settings
    training_days: int = 180
    validation_days: int = 30
    min_samples_for_training: int = 10000

    # Feature settings
    feature_version: str = "v1.0.0"

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def db_url(self) -> str:
        """Database connection string for SQLAlchemy"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
