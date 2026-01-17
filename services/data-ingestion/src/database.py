"""
Quantum Trading AI - Database Handler
======================================
Async database operations for storing candle data.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from collections import deque

import asyncpg
import redis.asyncio as redis

from .config import settings
from .collector import Candle


logger = logging.getLogger(__name__)


class DatabaseHandler:
    """
    Handles all database operations for the data ingestion service.

    Features:
    - Async connection pooling with asyncpg
    - Batch inserts for efficiency
    - Buffering with periodic flush
    - Redis caching for latest prices
    """

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[redis.Redis] = None

        # Buffer for batch inserts
        self.buffer: deque[Candle] = deque(maxlen=10000)
        self.buffer_lock = asyncio.Lock()

        # Stats
        self.candles_inserted = 0
        self.insert_errors = 0

    async def connect(self):
        """Initialize database and Redis connections"""
        # PostgreSQL connection pool
        logger.info(f"Connecting to PostgreSQL at {settings.db_host}:{settings.db_port}...")
        self.pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
            min_size=5,
            max_size=20,
            command_timeout=60,
        )
        logger.info("PostgreSQL connection established")

        # Redis connection
        logger.info(f"Connecting to Redis at {settings.redis_host}:{settings.redis_port}...")
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
        await self.redis.ping()
        logger.info("Redis connection established")

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
        if self.redis:
            await self.redis.close()
        logger.info("Database connections closed")

    async def add_candle(self, candle: Candle):
        """
        Add a candle to the buffer.
        Only closed candles are buffered for database insertion.
        All candles update Redis cache.
        """
        # Always update Redis with latest price
        await self._update_redis_cache(candle)

        # Only buffer closed candles for DB insertion
        if candle.is_closed:
            async with self.buffer_lock:
                self.buffer.append(candle)

    async def _update_redis_cache(self, candle: Candle):
        """Update Redis with latest price data"""
        if not self.redis:
            return

        try:
            key = f"price:{candle.symbol}"
            data = {
                "symbol": candle.symbol,
                "price": str(candle.close),
                "timestamp": candle.timestamp.isoformat(),
                "volume_24h": str(candle.volume),
            }
            await self.redis.hset(key, mapping=data)
            await self.redis.expire(key, 300)  # 5 minute TTL

            # Also maintain a sorted set of recent prices for quick lookup
            await self.redis.zadd(
                f"prices:{candle.symbol}",
                {str(candle.close): candle.timestamp.timestamp()}
            )
            # Keep only last 100 prices
            await self.redis.zremrangebyrank(f"prices:{candle.symbol}", 0, -101)

        except Exception as e:
            logger.error(f"Redis cache update error: {e}")

    async def flush_buffer(self) -> int:
        """
        Flush buffered candles to database.
        Returns number of candles inserted.
        """
        if not self.pool:
            logger.error("Database pool not initialized")
            return 0

        async with self.buffer_lock:
            if not self.buffer:
                return 0

            candles_to_insert = list(self.buffer)
            self.buffer.clear()

        if not candles_to_insert:
            return 0

        try:
            # Prepare data for batch insert
            records = [
                (
                    candle.timestamp,
                    candle.symbol,
                    "binance",  # exchange
                    settings.candle_interval,  # timeframe
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    candle.quote_volume,
                    candle.trades,
                )
                for candle in candles_to_insert
            ]

            async with self.pool.acquire() as conn:
                # Use executemany with ON CONFLICT for upsert
                await conn.executemany("""
                    INSERT INTO candles (
                        time, symbol, exchange, timeframe,
                        open, high, low, close,
                        volume, quote_volume, trades
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (time, symbol, exchange, timeframe)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        quote_volume = EXCLUDED.quote_volume,
                        trades = EXCLUDED.trades
                """, records)

                inserted = len(records)
                self.candles_inserted += inserted
                logger.info(f"Flushed {inserted} candles to database")
                return inserted

        except Exception as e:
            logger.error(f"Error flushing buffer to database: {e}")
            self.insert_errors += 1
            # Put candles back in buffer for retry
            async with self.buffer_lock:
                for candle in reversed(candles_to_insert):
                    self.buffer.appendleft(candle)
            return 0

    async def get_latest_candles(
        self,
        symbol: str,
        limit: int = 100,
        timeframe: str = "1m"
    ) -> List[Dict]:
        """Fetch latest candles from database"""
        if not self.pool:
            return []

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT time, open, high, low, close, volume, quote_volume, trades
                    FROM candles
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY time DESC
                    LIMIT $3
                """, symbol.upper(), timeframe, limit)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return []

    async def get_candle_count(self, symbol: Optional[str] = None) -> int:
        """Get total candle count (optionally filtered by symbol)"""
        if not self.pool:
            return 0

        try:
            async with self.pool.acquire() as conn:
                if symbol:
                    result = await conn.fetchval(
                        "SELECT COUNT(*) FROM candles WHERE symbol = $1",
                        symbol.upper()
                    )
                else:
                    result = await conn.fetchval("SELECT COUNT(*) FROM candles")
                return result or 0
        except Exception as e:
            logger.error(f"Error counting candles: {e}")
            return 0

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Redis cache"""
        if not self.redis:
            return None

        try:
            data = await self.redis.hget(f"price:{symbol.upper()}", "price")
            return float(data) if data else None
        except Exception as e:
            logger.error(f"Error getting price from Redis: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get database handler statistics"""
        return {
            "buffer_size": len(self.buffer),
            "candles_inserted": self.candles_inserted,
            "insert_errors": self.insert_errors,
            "pool_size": self.pool.get_size() if self.pool else 0,
            "pool_free": self.pool.get_idle_size() if self.pool else 0,
        }
