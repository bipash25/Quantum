"""Configuration for outcome tracker service."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # REQUIRED - must be set via environment variables (no defaults for secrets)
    database_url: str = os.environ["DATABASE_URL"]
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    telegram_token: str = os.environ["TELEGRAM_BOT_TOKEN"]
    telegram_channel_id: str = os.environ["TELEGRAM_CHANNEL_ID"]
    check_interval_seconds: int = int(os.getenv("CHECK_INTERVAL", "60"))


settings = Settings()
