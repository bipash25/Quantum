"""
Quantum Trading AI - Data Ingestion Service
============================================
Main entry point for real-time data collection from Binance.

This service:
1. Connects to Binance WebSocket for real-time kline data
2. Buffers incoming candles in memory
3. Periodically flushes to TimescaleDB
4. Maintains Redis cache for latest prices
5. Exposes Prometheus metrics for monitoring
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from prometheus_client import start_http_server, Counter, Gauge, Histogram

from .config import settings
from .collector import BinanceWebSocketCollector, BinanceRESTClient, Candle
from .database import DatabaseHandler


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================
CANDLES_RECEIVED = Counter(
    "data_ingestion_candles_received_total",
    "Total candles received from WebSocket",
    ["symbol"]
)
CANDLES_STORED = Counter(
    "data_ingestion_candles_stored_total",
    "Total candles stored in database",
)
WEBSOCKET_RECONNECTS = Counter(
    "data_ingestion_websocket_reconnects_total",
    "Number of WebSocket reconnections",
)
BUFFER_SIZE = Gauge(
    "data_ingestion_buffer_size",
    "Current buffer size (candles pending insertion)",
)
LATEST_PRICE = Gauge(
    "data_ingestion_latest_price",
    "Latest price for each symbol",
    ["symbol"]
)
FLUSH_DURATION = Histogram(
    "data_ingestion_flush_duration_seconds",
    "Time taken to flush buffer to database",
)


class DataIngestionService:
    """
    Main service class that orchestrates data collection.
    """

    def __init__(self):
        self.collector: BinanceWebSocketCollector | None = None
        self.rest_client: BinanceRESTClient | None = None
        self.db: DatabaseHandler | None = None
        self.running = False

        # Background tasks
        self.flush_task: asyncio.Task | None = None
        self.stats_task: asyncio.Task | None = None

    async def on_candle(self, candle: Candle):
        """Callback for each received candle"""
        # Update metrics
        CANDLES_RECEIVED.labels(symbol=candle.symbol).inc()
        LATEST_PRICE.labels(symbol=candle.symbol).set(candle.close)

        # Store in database handler
        if self.db:
            await self.db.add_candle(candle)
            BUFFER_SIZE.set(len(self.db.buffer))

    async def flush_loop(self):
        """Periodically flush buffer to database"""
        while self.running:
            try:
                await asyncio.sleep(settings.buffer_flush_interval)

                if self.db:
                    with FLUSH_DURATION.time():
                        count = await self.db.flush_buffer()
                    if count > 0:
                        CANDLES_STORED.inc(count)
                        BUFFER_SIZE.set(len(self.db.buffer))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    async def stats_loop(self):
        """Periodically log statistics"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Log every minute

                if self.collector and self.db:
                    collector_stats = self.collector.get_stats()
                    db_stats = self.db.get_stats()

                    logger.info(
                        f"Stats | "
                        f"Messages: {collector_stats['messages_received']} | "
                        f"Closed: {collector_stats['candles_closed']} | "
                        f"Buffer: {db_stats['buffer_size']} | "
                        f"Stored: {db_stats['candles_inserted']} | "
                        f"Errors: {collector_stats['errors']}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats loop: {e}")

    async def start(self):
        """Start the data ingestion service"""
        logger.info("=" * 60)
        logger.info("Quantum Trading AI - Data Ingestion Service")
        logger.info("=" * 60)
        logger.info(f"Symbols: {len(settings.symbols)}")
        logger.info(f"Interval: {settings.candle_interval}")
        logger.info(f"Flush interval: {settings.buffer_flush_interval}s")

        self.running = True

        # Start Prometheus metrics server
        logger.info("Starting Prometheus metrics server on port 8001...")
        start_http_server(8001)

        # Initialize database handler
        logger.info("Initializing database connections...")
        self.db = DatabaseHandler()
        await self.db.connect()

        # Initialize REST client
        self.rest_client = BinanceRESTClient()

        # Initialize WebSocket collector
        self.collector = BinanceWebSocketCollector(
            symbols=settings.symbols,
            interval=settings.candle_interval,
            on_candle=self.on_candle,
        )

        # Start background tasks
        self.flush_task = asyncio.create_task(self.flush_loop())
        self.stats_task = asyncio.create_task(self.stats_loop())

        # Start WebSocket collector (this runs indefinitely)
        logger.info("Starting WebSocket collector...")
        await self.collector.start()

    async def stop(self):
        """Stop the data ingestion service"""
        logger.info("Stopping data ingestion service...")
        self.running = False

        # Cancel background tasks
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass

        if self.stats_task:
            self.stats_task.cancel()
            try:
                await self.stats_task
            except asyncio.CancelledError:
                pass

        # Final flush
        if self.db:
            logger.info("Final buffer flush...")
            await self.db.flush_buffer()

        # Stop collector
        if self.collector:
            await self.collector.stop()

        # Close REST client
        if self.rest_client:
            await self.rest_client.close()

        # Close database connections
        if self.db:
            await self.db.close()

        logger.info("Data ingestion service stopped")


async def main():
    """Main entry point"""
    service = DataIngestionService()

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(service.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await service.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
