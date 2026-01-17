"""
Quantum Trading AI - Telegram Bot Configuration
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


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

    # Database (REQUIRED - must be set via environment variables)
    db_host: str = Field(default="timescaledb", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_user: str = Field(..., alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")
    db_name: str = Field(..., alias="DB_NAME")

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
