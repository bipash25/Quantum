#!/usr/bin/env python3
"""
Quantum Trading AI - Historical Data Backfill Script
=====================================================
Downloads historical OHLCV data from Binance for model training.

Usage:
    python scripts/backfill.py --days 180 --symbols BTCUSDT,ETHUSDT
    python scripts/backfill.py --days 365 --all  # All MVP symbols
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import List

import aiohttp
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Top 50 Crypto Symbols by Market Cap
TOP_50_SYMBOLS = [
    # Top 20 (original MVP)
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
    "FTMUSDT", "ALGOUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT",

    # Additional 30 (top 50 total)
    "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT",
    "TIAUSDT", "SEIUSDT", "RUNEUSDT", "RENDERUSDT", "WLDUSDT",
    "IMXUSDT", "LDOUSDT", "STXUSDT", "FILUSDT", "HBARUSDT",
    "VETUSDT", "ICPUSDT", "MKRUSDT", "QNTUSDT", "GRTUSDT",
    "FLOWUSDT", "XLMUSDT", "AXSUSDT", "THETAUSDT", "EGLDUSDT",
    "APEUSDT", "CHZUSDT", "EOSUSDT", "CFXUSDT", "ZILUSDT",
]

# Alias for backward compatibility
MVP_SYMBOLS = TOP_50_SYMBOLS

# Binance API
BINANCE_API = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

# Database config - REQUIRED environment variables (no hardcoded defaults for secrets)
import os
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "database": os.environ.get("DB_NAME", "quantum_trading"),
}


async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = 1000,
) -> List[List]:
    """Fetch klines from Binance API"""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit,
    }

    url = f"{BINANCE_API}{KLINES_ENDPOINT}"

    try:
        async with session.get(url, params=params) as response:
            if response.status == 429:
                # Rate limited, wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await fetch_klines(session, symbol, interval, start_time, end_time, limit)

            if response.status != 200:
                error = await response.text()
                logger.error(f"API error for {symbol}: {response.status} - {error}")
                return []

            return await response.json()

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return []


async def backfill_symbol(
    session: aiohttp.ClientSession,
    pool: asyncpg.Pool,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1m",
) -> int:
    """Backfill historical data for a single symbol"""
    logger.info(f"Backfilling {symbol} from {start_date.date()} to {end_date.date()}...")

    total_candles = 0
    current_start = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    # Calculate expected candles for progress tracking
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes_per_candle = interval_minutes.get(interval, 1)
    total_minutes = (end_date - start_date).total_seconds() / 60
    expected_candles = int(total_minutes / minutes_per_candle)

    while current_start < end_ms:
        klines = await fetch_klines(
            session, symbol, interval, current_start, end_ms, limit=1000
        )

        if not klines:
            break

        # Prepare records for batch insert
        records = []
        for k in klines:
            records.append((
                datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),  # time
                symbol,                                                  # symbol
                "binance",                                               # exchange
                interval,                                                # timeframe
                float(k[1]),                                             # open
                float(k[2]),                                             # high
                float(k[3]),                                             # low
                float(k[4]),                                             # close
                float(k[5]),                                             # volume
                float(k[7]),                                             # quote_volume
                int(k[8]),                                               # trades
            ))

        # Batch insert
        async with pool.acquire() as conn:
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

        total_candles += len(klines)
        progress = min(100, (total_candles / expected_candles) * 100)

        # Move to next batch
        last_time = klines[-1][0]
        current_start = last_time + 1

        # Rate limiting: max 1200 requests/min, be conservative
        await asyncio.sleep(0.1)

        if total_candles % 10000 == 0:
            logger.info(f"  {symbol}: {total_candles:,} candles ({progress:.1f}%)")

    logger.info(f"  {symbol}: Completed - {total_candles:,} candles")
    return total_candles


async def main(args):
    """Main backfill function"""
    logger.info("=" * 60)
    logger.info("Quantum Trading AI - Historical Data Backfill")
    logger.info("=" * 60)

    # Parse symbols
    if args.all:
        symbols = MVP_SYMBOLS
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = MVP_SYMBOLS

    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    logger.info("")

    # Connect to database
    logger.info("Connecting to database...")
    try:
        pool = await asyncpg.create_pool(**DB_CONFIG, min_size=5, max_size=20)
    except OSError as e:
        # Fallback to localhost if hostname resolution fails (common when running outside docker)
        if DB_CONFIG["host"] not in ("localhost", "127.0.0.1"):
            logger.warning(f"Failed to connect to {DB_CONFIG['host']}: {e}")
            logger.info("Attempting fallback to localhost...")
            DB_CONFIG["host"] = "localhost"
            pool = await asyncpg.create_pool(**DB_CONFIG, min_size=5, max_size=20)
        else:
            raise e

    # Create HTTP session
    async with aiohttp.ClientSession() as session:
        total_candles = 0

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")

            try:
                count = await backfill_symbol(
                    session, pool, symbol, start_date, end_date, args.interval
                )
                total_candles += count
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

            # Small delay between symbols
            await asyncio.sleep(0.5)

    await pool.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Backfill complete! Total candles: {total_candles:,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill historical OHLCV data from Binance"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3650,
        help="Number of days to backfill (default: 180)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backfill all MVP symbols"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Candle interval (default: 1m)"
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Backfill cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
