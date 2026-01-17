"""
Quantum Trading AI - Shared Utilities
=====================================
Common utility functions used across services.
"""
import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from functools import wraps
import time

import redis
import asyncpg


logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

class DatabasePool:
    """Async PostgreSQL connection pool manager"""

    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def get_pool(cls, config: Dict[str, Any]) -> asyncpg.Pool:
        """Get or create the database connection pool"""
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                host=config["host"],
                port=config["port"],
                user=config["user"],
                password=config["password"],
                database=config["database"],
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
        return cls._pool

    @classmethod
    async def close_pool(cls):
        """Close the connection pool"""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None


class RedisClient:
    """Redis client wrapper with connection management"""

    _client: Optional[redis.Redis] = None

    @classmethod
    def get_client(cls, config: Dict[str, Any]) -> redis.Redis:
        """Get or create Redis client"""
        if cls._client is None:
            cls._client = redis.Redis(
                host=config["host"],
                port=config["port"],
                db=config.get("db", 0),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
        return cls._client

    @classmethod
    def close_client(cls):
        """Close Redis connection"""
        if cls._client:
            cls._client.close()
            cls._client = None


# =============================================================================
# TIME UTILITIES
# =============================================================================

def utc_now() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


def timestamp_ms() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime"""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp"""
    return int(dt.timestamp() * 1000)


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes"""
    mapping = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
    }
    return mapping.get(timeframe, 1)


# =============================================================================
# RETRY UTILITIES
# =============================================================================

def retry_async(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),  # type: ignore
):
    """Decorator for async retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * backoff_multiplier, max_delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception
        return wrapper
    return decorator


def retry_sync(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),  # type: ignore
):
    """Decorator for sync retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_multiplier, max_delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_candle(candle: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate OHLCV candle data.
    Returns (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    required = ["open", "high", "low", "close", "volume"]
    for field in required:
        if field not in candle:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    o, h, l, c, v = candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]

    # OHLC integrity checks
    if h < l:
        errors.append(f"High ({h}) is less than Low ({l})")

    if h < max(o, c):
        errors.append(f"High ({h}) is less than max of Open/Close ({max(o, c)})")

    if l > min(o, c):
        errors.append(f"Low ({l}) is greater than min of Open/Close ({min(o, c)})")

    if v < 0:
        errors.append(f"Negative volume: {v}")

    # Price sanity checks
    for name, value in [("open", o), ("high", h), ("low", l), ("close", c)]:
        if value <= 0:
            errors.append(f"Non-positive {name}: {value}")

    return len(errors) == 0, errors


def is_price_outlier(
    current: float,
    previous: float,
    max_change_pct: float = 50.0
) -> bool:
    """Check if price change is an outlier (potential bad data)"""
    if previous <= 0:
        return False
    change_pct = abs((current - previous) / previous) * 100
    return change_pct > max_change_pct


# =============================================================================
# HASHING UTILITIES
# =============================================================================

def generate_signal_id(
    symbol: str,
    timeframe: str,
    timestamp: datetime,
    direction: str
) -> str:
    """Generate unique signal ID"""
    data = f"{symbol}_{timeframe}_{timestamp.isoformat()}_{direction}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def generate_model_version(
    model_type: str,
    asset_class: str,
    timeframe: str,
    timestamp: Optional[datetime] = None
) -> str:
    """Generate model version string"""
    if timestamp is None:
        timestamp = utc_now()
    date_str = timestamp.strftime("%Y%m%d")
    return f"{model_type}_{asset_class}_{timeframe}_{date_str}"


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_price(price: float, decimals: int = 4) -> str:
    """Format price with appropriate precision"""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    else:
        return f"{price:.8f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value"""
    return f"{value:+.{decimals}f}%"


def format_large_number(value: float) -> str:
    """Format large numbers with K, M, B suffixes"""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Simple circuit breaker for external service calls.

    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Too many failures, calls fail fast
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"

    def record_success(self):
        """Record a successful call"""
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record a failed call"""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker {self.name} OPENED after {self.failures} failures")

    def can_execute(self) -> bool:
        """Check if a call can be executed"""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                return True
            return False

        # HALF_OPEN: allow one test call
        return True

    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self.state
