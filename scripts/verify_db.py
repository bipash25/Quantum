import asyncio
import os
import asyncpg
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "database": os.environ.get("DB_NAME", "quantum_trading"),
}

async def verify_db():
    print("Connecting to database...")
    try:
        pool = await asyncpg.create_pool(**DB_CONFIG)
    except Exception as e:
        print(f"Connection failed: {e}")
        # Try localhost fallback as per backfill script
        DB_CONFIG["host"] = "localhost"
        try:
            pool = await asyncpg.create_pool(**DB_CONFIG)
        except Exception as e:
            print(f"Fallback failed: {e}")
            return

    async with pool.acquire() as conn:
        print("\n--- Database Verification ---")

        # 1. Total Count
        total = await conn.fetchval("SELECT COUNT(*) FROM candles")
        print(f"Total Candles: {total:,}")

        # 2. Per Symbol Stats
        print(f"\n{'Symbol':<10} | {'Count':<10} | {'Start Date':<12} | {'End Date':<12}")
        print("-" * 52)

        rows = await conn.fetch("""
            SELECT
                symbol,
                COUNT(*) as count,
                MIN(time) as start_time,
                MAX(time) as end_time
            FROM candles
            GROUP BY symbol
            ORDER BY count DESC
        """)

        for r in rows:
            symbol = r['symbol']
            count = r['count']
            start = r['start_time'].strftime('%Y-%m-%d') if r['start_time'] else "N/A"
            end = r['end_time'].strftime('%Y-%m-%d') if r['end_time'] else "N/A"
            print(f"{symbol:<10} | {count:<10,} | {start:<12} | {end:<12}")

    await pool.close()

if __name__ == "__main__":
    asyncio.run(verify_db())
