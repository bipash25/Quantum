"""Configuration for API service."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # REQUIRED - must be set via environment variables (no defaults for secrets)
    database_url: str = os.environ["DATABASE_URL"]
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    api_key_header: str = "X-API-Key"
    jwt_secret: str = os.environ["JWT_SECRET"]
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24


settings = Settings()
