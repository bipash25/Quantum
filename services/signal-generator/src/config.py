"""
Quantum Trading AI - Signal Generator Configuration
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuration for signal generator service"""

    # Database (REQUIRED - must be set via environment variables)
    db_host: str = Field(..., alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_user: str = Field(..., alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")
    db_name: str = Field(..., alias="DB_NAME")

    # Redis
    redis_host: str = Field(default="redis", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")

    # Model directory
    model_dir: str = Field(default="/app/models", alias="MODEL_DIR")

    # Signal settings
    min_confidence: float = Field(default=0.55, alias="MIN_CONFIDENCE")
    min_volume_ratio: float = Field(default=0.5, alias="MIN_VOLUME_RATIO")
    max_signals_per_run: int = Field(default=10, alias="MAX_SIGNALS_PER_RUN")

    # Timeframe (4h or 1d)
    signal_timeframe: str = Field(default="4h", alias="SIGNAL_TIMEFRAME")

    # Generation interval in seconds
    generation_interval: int = Field(default=300, alias="GENERATION_INTERVAL")

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
