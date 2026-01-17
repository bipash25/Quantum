"""
Quantum Trading AI - Binance WebSocket Collector
=================================================
Real-time data collection from Binance via WebSocket.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

import websockets
from websockets.exceptions import ConnectionClosed
import aiohttp

from .config import settings


logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLCV candlestick data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    is_closed: bool

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
            "is_closed": self.is_closed,
        }


@dataclass
class CollectorStats:
    """Statistics for monitoring"""
    messages_received: int = 0
    candles_closed: int = 0
    errors: int = 0
    reconnections: int = 0
    last_message_time: Optional[datetime] = None
    symbols_active: int = 0


class BinanceWebSocketCollector:
    """
    Collects real-time kline (candlestick) data from Binance WebSocket.

    Features:
    - Multi-symbol subscription
    - Automatic reconnection with exponential backoff
    - Message validation
    - Callback-based architecture for flexibility
    """

    def __init__(
        self,
        symbols: List[str],
        interval: str = "1m",
        on_candle: Optional[Callable[[Candle], None]] = None,
    ):
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.on_candle = on_candle

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self.stats = CollectorStats()

        # Connection settings
        self.max_reconnect_attempts = 10
        self.base_reconnect_delay = 1.0
        self.max_reconnect_delay = 60.0

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL for multiple streams"""
        # For combined streams, use the /stream endpoint
        streams = [f"{symbol}@kline_{self.interval}" for symbol in self.symbols]
        streams_param = "/".join(streams)
        return f"{settings.binance_ws_url}/stream?streams={streams_param}"

    def _parse_kline_message(self, data: Dict) -> Optional[Candle]:
        """Parse a kline WebSocket message into a Candle object"""
        try:
            # Combined stream format: {"stream": "btcusdt@kline_1m", "data": {...}}
            if "data" in data:
                data = data["data"]

            if data.get("e") != "kline":
                return None

            kline = data["k"]

            return Candle(
                symbol=kline["s"],  # Symbol
                timestamp=datetime.fromtimestamp(kline["t"] / 1000, tz=timezone.utc),
                open=float(kline["o"]),
                high=float(kline["h"]),
                low=float(kline["l"]),
                close=float(kline["c"]),
                volume=float(kline["v"]),
                quote_volume=float(kline["q"]),
                trades=int(kline["n"]),
                is_closed=kline["x"],  # Is this kline closed?
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing kline message: {e}")
            self.stats.errors += 1
            return None

    def _validate_candle(self, candle: Candle) -> bool:
        """Validate candle data integrity"""
        if candle.high < candle.low:
            logger.warning(f"Invalid candle: high < low for {candle.symbol}")
            return False
        if candle.high < max(candle.open, candle.close):
            logger.warning(f"Invalid candle: high < open/close for {candle.symbol}")
            return False
        if candle.low > min(candle.open, candle.close):
            logger.warning(f"Invalid candle: low > open/close for {candle.symbol}")
            return False
        if candle.volume < 0:
            logger.warning(f"Invalid candle: negative volume for {candle.symbol}")
            return False
        return True

    async def _handle_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            candle = self._parse_kline_message(data)

            if candle is None:
                return

            self.stats.messages_received += 1
            self.stats.last_message_time = datetime.now(timezone.utc)

            if not self._validate_candle(candle):
                self.stats.errors += 1
                return

            if candle.is_closed:
                self.stats.candles_closed += 1
                logger.debug(f"Closed candle: {candle.symbol} @ {candle.close:.4f}")

            # Call the callback with the candle
            if self.on_candle:
                await self._invoke_callback(candle)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            self.stats.errors += 1

    async def _invoke_callback(self, candle: Candle):
        """Invoke the on_candle callback (handles both sync and async)"""
        if self.on_candle is None:
            return

        try:
            if asyncio.iscoroutinefunction(self.on_candle):
                await self.on_candle(candle)
            else:
                self.on_candle(candle)
        except Exception as e:
            logger.error(f"Error in candle callback: {e}")
            self.stats.errors += 1

    async def connect(self):
        """Establish WebSocket connection with reconnection logic"""
        reconnect_attempt = 0

        while self.running and reconnect_attempt < self.max_reconnect_attempts:
            try:
                url = self._build_ws_url()
                logger.info(f"Connecting to Binance WebSocket ({len(self.symbols)} symbols)...")

                async with websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self.ws = ws
                    self.stats.symbols_active = len(self.symbols)
                    reconnect_attempt = 0  # Reset on successful connection

                    logger.info(f"Connected! Receiving data for {len(self.symbols)} symbols")

                    async for message in ws:
                        if not self.running:
                            break
                        await self._handle_message(message)

            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self.stats.reconnections += 1

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.stats.errors += 1
                self.stats.reconnections += 1

            if self.running:
                # Exponential backoff
                delay = min(
                    self.base_reconnect_delay * (2 ** reconnect_attempt),
                    self.max_reconnect_delay
                )
                logger.info(f"Reconnecting in {delay:.1f}s (attempt {reconnect_attempt + 1})")
                await asyncio.sleep(delay)
                reconnect_attempt += 1

        if reconnect_attempt >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached!")

    async def start(self):
        """Start the WebSocket collector"""
        self.running = True
        await self.connect()

    async def stop(self):
        """Stop the WebSocket collector"""
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        logger.info("WebSocket collector stopped")

    def get_stats(self) -> Dict:
        """Get collector statistics"""
        # Check connection state
        is_connected = False
        if self.ws is not None:
            try:
                is_connected = self.ws.state.name == "OPEN"
            except Exception:
                is_connected = False

        return {
            "messages_received": self.stats.messages_received,
            "candles_closed": self.stats.candles_closed,
            "errors": self.stats.errors,
            "reconnections": self.stats.reconnections,
            "last_message_time": (
                self.stats.last_message_time.isoformat()
                if self.stats.last_message_time else None
            ),
            "symbols_active": self.stats.symbols_active,
            "is_connected": is_connected,
        }


class BinanceRESTClient:
    """
    REST API client for Binance.
    Used for:
    - Historical data backfill
    - Fallback when WebSocket fails
    - Initial data fetch on startup
    """

    def __init__(self):
        self.base_url = settings.binance_rest_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Candle]:
        """
        Fetch historical klines (candlestick) data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1m", "1h", "1d")
            limit: Number of klines to fetch (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of Candle objects
        """
        await self._ensure_session()

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        url = f"{self.base_url}/api/v3/klines"

        try:
            async with self.session.get(url, params=params) as response:  # type: ignore
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Binance API error: {response.status} - {error_text}")
                    return []

                data = await response.json()

                candles = []
                for kline in data:
                    candle = Candle(
                        symbol=symbol.upper(),
                        timestamp=datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=float(kline[4]),
                        volume=float(kline[5]),
                        quote_volume=float(kline[7]),
                        trades=int(kline[8]),
                        is_closed=True,  # Historical data is always closed
                    )
                    candles.append(candle)

                return candles

        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return []

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        await self._ensure_session()

        url = f"{self.base_url}/api/v3/ticker/price"
        params = {"symbol": symbol.upper()}

        try:
            async with self.session.get(url, params=params) as response:  # type: ignore
                if response.status == 200:
                    data = await response.json()
                    return float(data["price"])
                return None
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    async def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules"""
        await self._ensure_session()

        url = f"{self.base_url}/api/v3/exchangeInfo"

        try:
            async with self.session.get(url) as response:  # type: ignore
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {}
