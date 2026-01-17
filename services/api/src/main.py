"""
Quantum Trading AI - REST API Service
======================================
Provides API access for Pro tier users to get signals programmatically.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import logging
import os

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
import jwt

from .config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quantum Trading AI API",
    description="AI-powered crypto trading signals API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
engine = create_engine(settings.database_url)


# =============================================================================
# MODELS
# =============================================================================

class SignalResponse(BaseModel):
    id: int
    symbol: str
    direction: str
    timeframe: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]
    risk_reward_ratio: float
    status: str
    reasoning: Optional[str]
    created_at: datetime
    valid_until: datetime

class SignalListResponse(BaseModel):
    signals: List[SignalResponse]
    total: int
    page: int
    limit: int

class OutcomeResponse(BaseModel):
    signal_id: int
    outcome: str
    exit_price: Optional[float]
    pnl_percent: Optional[float]
    duration_minutes: Optional[int]

class PerformanceResponse(BaseModel):
    total_signals: int
    wins: int
    losses: int
    expired: int
    pending: int
    win_rate: float
    avg_pnl: float
    total_pnl: float

class SymbolStats(BaseModel):
    symbol: str
    total_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database: str
    signals_active: int


# =============================================================================
# AUTH
# =============================================================================

# API keys loaded from environment variable (JSON format)
# Example: API_KEYS='{"key1": {"tier": "free", "rate_limit": 10}}'
import json
_api_keys_json = os.environ.get("API_KEYS", "{}")
VALID_API_KEYS = json.loads(_api_keys_json) if _api_keys_json else {}

async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header."""
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return VALID_API_KEYS[x_api_key]


# =============================================================================
# ENDPOINTS
# =============================================================================

# -----------------------------------------------------------------------------
# PUBLIC ENDPOINTS (No auth required - for landing page)
# -----------------------------------------------------------------------------

class PublicStatsResponse(BaseModel):
    total_signals: int
    active_signals: int
    completed_signals: int
    win_count: int
    loss_count: int
    win_rate: Optional[float]
    avg_confidence: float
    symbols_tracked: int
    last_signal_time: Optional[datetime]

class PublicSignalPreview(BaseModel):
    symbol: str
    direction: str
    timeframe: str
    confidence: float
    created_at: datetime
    status: str

class PublicRecentSignals(BaseModel):
    signals: List[PublicSignalPreview]

@app.get("/public/stats", response_model=PublicStatsResponse)
async def get_public_stats():
    """Public stats for landing page - no authentication required."""
    try:
        with engine.connect() as conn:
            # Get total signals
            total = conn.execute(text("SELECT COUNT(*) FROM signals")).scalar() or 0

            # Get active signals
            active = conn.execute(text("SELECT COUNT(*) FROM signals WHERE status = 'active'")).scalar() or 0

            # Get completed signals (hit_tp or hit_sl)
            completed = conn.execute(text(
                "SELECT COUNT(*) FROM signals WHERE status IN ('hit_tp', 'hit_sl', 'expired')"
            )).scalar() or 0

            # Get win/loss counts
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE status = 'hit_tp'")).scalar() or 0
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE status = 'hit_sl'")).scalar() or 0

            # Calculate win rate (only if we have completed trades)
            total_completed = wins + losses
            win_rate = (wins / total_completed * 100) if total_completed > 0 else None

            # Get average confidence
            avg_conf = conn.execute(text("SELECT AVG(confidence) FROM signals")).scalar() or 0

            # Get symbols count
            symbols = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM signals")).scalar() or 16

            # Get last signal time
            last_time = conn.execute(text(
                "SELECT created_at FROM signals ORDER BY created_at DESC LIMIT 1"
            )).scalar()

        return PublicStatsResponse(
            total_signals=total,
            active_signals=active,
            completed_signals=completed,
            win_count=wins,
            loss_count=losses,
            win_rate=round(win_rate, 1) if win_rate else None,
            avg_confidence=round(avg_conf, 1),
            symbols_tracked=symbols if symbols > 0 else 16,
            last_signal_time=last_time,
        )
    except Exception as e:
        logger.error(f"Error getting public stats: {e}")
        # Return default stats on error
        return PublicStatsResponse(
            total_signals=0,
            active_signals=0,
            completed_signals=0,
            win_count=0,
            loss_count=0,
            win_rate=None,
            avg_confidence=0,
            symbols_tracked=16,
            last_signal_time=None,
        )

@app.get("/public/recent", response_model=PublicRecentSignals)
async def get_public_recent_signals():
    """Get recent signals preview for landing page - sanitized, no sensitive data."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT symbol, direction, timeframe, confidence, created_at, status
                FROM signals
                ORDER BY created_at DESC
                LIMIT 3
            """))
            rows = result.fetchall()

        signals = [
            PublicSignalPreview(
                symbol=row[0],
                direction=row[1],
                timeframe=row[2],
                confidence=round(row[3], 1),
                created_at=row[4],
                status=row[5],
            )
            for row in rows
        ]

        return PublicRecentSignals(signals=signals)
    except Exception as e:
        logger.error(f"Error getting recent signals: {e}")
        return PublicRecentSignals(signals=[])


# -----------------------------------------------------------------------------
# AUTHENTICATED ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM signals WHERE status = 'active'"))
            active_count = result.scalar()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
        active_count = 0

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        database=db_status,
        signals_active=active_count,
    )


@app.get("/signals", response_model=SignalListResponse)
async def get_signals(
    auth: dict = Depends(verify_api_key),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    direction: Optional[str] = Query(None, description="Filter by direction (LONG/SHORT)"),
    status: Optional[str] = Query("active", description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
):
    """Get trading signals with optional filters."""
    offset = (page - 1) * limit

    query = """
        SELECT
            id, symbol, direction, timeframe, confidence,
            entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
            risk_reward_ratio, status, reasoning, created_at, valid_until
        FROM signals
        WHERE 1=1
    """
    params = {}

    if symbol:
        query += " AND symbol = :symbol"
        params["symbol"] = symbol.upper()

    if direction:
        query += " AND direction = :direction"
        params["direction"] = direction.upper()

    if status:
        query += " AND status = :status"
        params["status"] = status

    # Count total
    count_query = f"SELECT COUNT(*) FROM ({query}) AS subq"

    query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
    params["limit"] = limit
    params["offset"] = offset

    with engine.connect() as conn:
        total = conn.execute(text(count_query), params).scalar()
        result = conn.execute(text(query), params)
        rows = result.fetchall()

    signals = [
        SignalResponse(
            id=row[0],
            symbol=row[1],
            direction=row[2],
            timeframe=row[3],
            confidence=row[4],
            entry_price=row[5],
            stop_loss=row[6],
            take_profit_1=row[7],
            take_profit_2=row[8],
            take_profit_3=row[9],
            risk_reward_ratio=row[10],
            status=row[11],
            reasoning=row[12],
            created_at=row[13],
            valid_until=row[14],
        )
        for row in rows
    ]

    return SignalListResponse(
        signals=signals,
        total=total,
        page=page,
        limit=limit,
    )


@app.get("/signals/{signal_id}", response_model=SignalResponse)
async def get_signal(
    signal_id: int,
    auth: dict = Depends(verify_api_key),
):
    """Get a specific signal by ID."""
    query = text("""
        SELECT
            id, symbol, direction, timeframe, confidence,
            entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
            risk_reward_ratio, status, reasoning, created_at, valid_until
        FROM signals
        WHERE id = :signal_id
        ORDER BY created_at DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"signal_id": signal_id})
        row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Signal not found")

    return SignalResponse(
        id=row[0],
        symbol=row[1],
        direction=row[2],
        timeframe=row[3],
        confidence=row[4],
        entry_price=row[5],
        stop_loss=row[6],
        take_profit_1=row[7],
        take_profit_2=row[8],
        take_profit_3=row[9],
        risk_reward_ratio=row[10],
        status=row[11],
        reasoning=row[12],
        created_at=row[13],
        valid_until=row[14],
    )


@app.get("/signals/{signal_id}/outcome", response_model=OutcomeResponse)
async def get_signal_outcome(
    signal_id: int,
    auth: dict = Depends(verify_api_key),
):
    """Get outcome for a specific signal."""
    query = text("""
        SELECT signal_id, outcome, exit_price, pnl_percent, duration_minutes
        FROM signal_outcomes
        WHERE signal_id = :signal_id
        ORDER BY created_at DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"signal_id": signal_id})
        row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Outcome not found")

    return OutcomeResponse(
        signal_id=row[0],
        outcome=row[1],
        exit_price=row[2],
        pnl_percent=row[3],
        duration_minutes=row[4],
    )


@app.get("/performance", response_model=PerformanceResponse)
async def get_performance(
    auth: dict = Depends(verify_api_key),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
):
    """Get overall performance statistics."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    query = """
        SELECT
            o.outcome,
            COUNT(*) as count,
            AVG(o.pnl_percent) as avg_pnl,
            SUM(o.pnl_percent) as total_pnl
        FROM signal_outcomes o
        JOIN signals s ON o.signal_id = s.id AND o.signal_created_at = s.created_at
        WHERE s.created_at >= :since
    """
    params = {"since": since}

    if symbol:
        query += " AND s.symbol = :symbol"
        params["symbol"] = symbol.upper()

    query += " GROUP BY o.outcome"

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = result.fetchall()

    stats = {row[0]: {"count": row[1], "avg_pnl": row[2] or 0, "total_pnl": row[3] or 0} for row in rows}

    wins = stats.get("win", {}).get("count", 0)
    losses = stats.get("loss", {}).get("count", 0)
    expired = stats.get("expired", {}).get("count", 0)
    pending = stats.get("pending", {}).get("count", 0)
    total = wins + losses + expired + pending

    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    total_pnl = sum(s.get("total_pnl", 0) for s in stats.values())
    avg_pnl = total_pnl / total if total > 0 else 0

    return PerformanceResponse(
        total_signals=total,
        wins=wins,
        losses=losses,
        expired=expired,
        pending=pending,
        win_rate=round(win_rate, 2),
        avg_pnl=round(avg_pnl, 2),
        total_pnl=round(total_pnl, 2),
    )


@app.get("/performance/by-symbol", response_model=List[SymbolStats])
async def get_performance_by_symbol(
    auth: dict = Depends(verify_api_key),
    days: int = Query(30, ge=1, le=365),
):
    """Get performance statistics grouped by symbol."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    query = text("""
        SELECT
            s.symbol,
            COUNT(*) as total,
            SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN o.outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            AVG(o.pnl_percent) as avg_pnl
        FROM signal_outcomes o
        JOIN signals s ON o.signal_id = s.id AND o.signal_created_at = s.created_at
        WHERE s.created_at >= :since
        GROUP BY s.symbol
        ORDER BY wins DESC
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"since": since})
        rows = result.fetchall()

    return [
        SymbolStats(
            symbol=row[0],
            total_signals=row[1],
            wins=row[2],
            losses=row[3],
            win_rate=round((row[2] / (row[2] + row[3]) * 100) if (row[2] + row[3]) > 0 else 0, 2),
            avg_pnl=round(row[4] or 0, 2),
        )
        for row in rows
    ]


@app.get("/symbols")
async def get_symbols(auth: dict = Depends(verify_api_key)):
    """Get list of tracked symbols."""
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT",
        "ALGOUSDT", "AAVEUSDT",
    ]
    return {"symbols": symbols, "count": len(symbols)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
