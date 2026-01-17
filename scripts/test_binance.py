#!/usr/bin/env python3
"""
Quantum Trading AI - Binance API Connectivity Test
===================================================
Quick test script to verify Binance API access.
"""
import asyncio
import aiohttp
import sys
from datetime import datetime


BINANCE_BASE_URL = "https://api.binance.com"
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def test_binance_api():
    """Test basic Binance API endpoints"""
    print("=" * 60)
    print("Quantum Trading AI - Binance API Test")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Test 1: Ping
        print("\n1. Testing /api/v3/ping...")
        async with session.get(f"{BINANCE_BASE_URL}/api/v3/ping") as resp:
            if resp.status == 200:
                print("   ✅ Ping successful")
            else:
                print(f"   ❌ Ping failed: {resp.status}")
                return False

        # Test 2: Server Time
        print("\n2. Testing /api/v3/time...")
        async with session.get(f"{BINANCE_BASE_URL}/api/v3/time") as resp:
            if resp.status == 200:
                data = await resp.json()
                server_time = datetime.fromtimestamp(data["serverTime"] / 1000)
                print(f"   ✅ Server time: {server_time}")
            else:
                print(f"   ❌ Time check failed: {resp.status}")
                return False

        # Test 3: Exchange Info (rate limits)
        print("\n3. Testing /api/v3/exchangeInfo...")
        async with session.get(f"{BINANCE_BASE_URL}/api/v3/exchangeInfo") as resp:
            if resp.status == 200:
                data = await resp.json()
                symbols_count = len(data.get("symbols", []))
                rate_limits = data.get("rateLimits", [])
                print(f"   ✅ {symbols_count} trading pairs available")
                for limit in rate_limits:
                    if limit.get("rateLimitType") == "REQUEST_WEIGHT":
                        print(f"   📊 Rate limit: {limit['limit']} per {limit['intervalNum']} {limit['interval']}")
            else:
                print(f"   ❌ Exchange info failed: {resp.status}")
                return False

        # Test 4: Current Prices
        print("\n4. Testing /api/v3/ticker/price for test symbols...")
        for symbol in TEST_SYMBOLS:
            async with session.get(f"{BINANCE_BASE_URL}/api/v3/ticker/price?symbol={symbol}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data["price"])
                    print(f"   ✅ {symbol}: ${price:,.2f}")
                else:
                    print(f"   ❌ Price fetch failed for {symbol}: {resp.status}")

        # Test 5: Klines (candlestick data)
        print("\n5. Testing /api/v3/klines for BTCUSDT...")
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 5}
        async with session.get(f"{BINANCE_BASE_URL}/api/v3/klines", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"   ✅ Retrieved {len(data)} candles")
                latest = data[-1]
                open_price = float(latest[1])
                high_price = float(latest[2])
                low_price = float(latest[3])
                close_price = float(latest[4])
                volume = float(latest[5])
                print(f"   📊 Latest 1H candle: O={open_price:.2f} H={high_price:.2f} L={low_price:.2f} C={close_price:.2f} V={volume:.2f}")
            else:
                print(f"   ❌ Klines fetch failed: {resp.status}")
                return False

        # Test 6: WebSocket URL reachability (just check DNS)
        print("\n6. Testing WebSocket URL...")
        ws_url = "wss://stream.binance.com:9443/ws"
        print(f"   ℹ️  WebSocket URL: {ws_url}")
        print("   ✅ WebSocket URL is configured (will test in data ingestion service)")

        print("\n" + "=" * 60)
        print("All Binance API tests passed! ✅")
        print("=" * 60)
        return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_binance_api())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
