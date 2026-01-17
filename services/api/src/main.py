"""
Quantum Trading AI - REST API Service
======================================
Provides API access for Pro tier users to get signals programmatically.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
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
# DASHBOARD ENDPOINTS (Public - for stats page)
# -----------------------------------------------------------------------------

class DashboardStats(BaseModel):
    total_signals: int
    wins: int
    losses: int
    expired: int
    active: int
    partial: int = 0  # Signals with partial TP fills
    win_rate: Optional[float]
    avg_rr_ratio: float
    avg_confidence: float
    best_symbol: Optional[str]
    worst_symbol: Optional[str]
    total_4h_signals: int
    total_24h_signals: int
    # P&L metrics
    cumulative_pnl: float = 0.0  # Total P&L with 2% risk per trade
    avg_pnl_per_trade: float = 0.0
    tp1_hits: int = 0
    tp2_hits: int = 0
    tp3_hits: int = 0
    last_updated: datetime

class SymbolPerformance(BaseModel):
    symbol: str
    total: int
    wins: int
    losses: int
    win_rate: Optional[float]
    avg_confidence: float

class DailyPerformance(BaseModel):
    date: str
    signals: int
    wins: int
    losses: int
    win_rate: Optional[float]

class RecentSignalDetail(BaseModel):
    id: int
    symbol: str
    direction: str
    timeframe: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    risk_reward_ratio: float
    status: str
    created_at: datetime

@app.get("/public/dashboard", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get comprehensive dashboard statistics including P&L."""
    try:
        with engine.connect() as conn:
            # Total signals
            total = conn.execute(text("SELECT COUNT(*) FROM signals")).scalar() or 0

            # Status breakdown (now including partial fills)
            status_query = conn.execute(text("""
                SELECT status, COUNT(*) FROM signals GROUP BY status
            """))
            status_counts = {row[0]: row[1] for row in status_query.fetchall()}

            wins = status_counts.get('hit_tp', 0) + status_counts.get('hit_tp3', 0)
            losses = status_counts.get('hit_sl', 0)
            expired = status_counts.get('expired', 0)
            active = status_counts.get('active', 0)
            partial = status_counts.get('hit_tp1', 0) + status_counts.get('hit_tp2', 0)

            # Win rate
            completed = wins + losses
            win_rate = (wins / completed * 100) if completed > 0 else None

            # Average R:R ratio
            avg_rr = conn.execute(text(
                "SELECT AVG(risk_reward_ratio) FROM signals"
            )).scalar() or 0

            # Average confidence
            avg_conf = conn.execute(text(
                "SELECT AVG(confidence) FROM signals"
            )).scalar() or 0

            # P&L from outcomes table
            pnl_stats = conn.execute(text("""
                SELECT
                    COALESCE(SUM(actual_pnl_percent), 0) as total_pnl,
                    COALESCE(AVG(actual_pnl_percent), 0) as avg_pnl,
                    COUNT(CASE WHEN tp1_hit THEN 1 END) as tp1_hits,
                    COUNT(CASE WHEN tp2_hit THEN 1 END) as tp2_hits,
                    COUNT(CASE WHEN tp3_hit THEN 1 END) as tp3_hits
                FROM signal_outcomes
                WHERE outcome IN ('win', 'loss', 'partial', 'expired')
            """)).fetchone()

            cumulative_pnl = pnl_stats[0] if pnl_stats else 0
            avg_pnl = pnl_stats[1] if pnl_stats else 0
            tp1_hits = pnl_stats[2] if pnl_stats else 0
            tp2_hits = pnl_stats[3] if pnl_stats else 0
            tp3_hits = pnl_stats[4] if pnl_stats else 0

            # Best/worst symbol by win rate
            symbol_stats = conn.execute(text("""
                SELECT symbol,
                       COUNT(*) as total,
                       SUM(CASE WHEN status IN ('hit_tp', 'hit_tp3') THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN status = 'hit_sl' THEN 1 ELSE 0 END) as losses
                FROM signals
                WHERE status IN ('hit_tp', 'hit_tp3', 'hit_sl')
                GROUP BY symbol
                HAVING COUNT(*) >= 3
                ORDER BY SUM(CASE WHEN status IN ('hit_tp', 'hit_tp3') THEN 1 ELSE 0 END)::float /
                         NULLIF(COUNT(*), 0) DESC
            """)).fetchall()

            best_symbol = symbol_stats[0][0] if symbol_stats else None
            worst_symbol = symbol_stats[-1][0] if len(symbol_stats) > 1 else None

            # Timeframe breakdown
            tf_query = conn.execute(text("""
                SELECT timeframe, COUNT(*) FROM signals GROUP BY timeframe
            """))
            tf_counts = {row[0]: row[1] for row in tf_query.fetchall()}

            total_4h = tf_counts.get('4h', 0)
            total_24h = tf_counts.get('1d', 0)

        return DashboardStats(
            total_signals=total,
            wins=wins,
            losses=losses,
            expired=expired,
            active=active,
            partial=partial,
            win_rate=round(win_rate, 1) if win_rate else None,
            avg_rr_ratio=round(avg_rr, 2),
            avg_confidence=round(avg_conf, 1),
            best_symbol=best_symbol,
            worst_symbol=worst_symbol,
            total_4h_signals=total_4h,
            total_24h_signals=total_24h,
            cumulative_pnl=round(cumulative_pnl, 2),
            avg_pnl_per_trade=round(avg_pnl, 2),
            tp1_hits=tp1_hits,
            tp2_hits=tp2_hits,
            tp3_hits=tp3_hits,
            last_updated=datetime.now(timezone.utc),
        )
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        return DashboardStats(
            total_signals=0, wins=0, losses=0, expired=0, active=0, partial=0,
            win_rate=None, avg_rr_ratio=0, avg_confidence=0,
            best_symbol=None, worst_symbol=None,
            total_4h_signals=0, total_24h_signals=0,
            cumulative_pnl=0, avg_pnl_per_trade=0,
            tp1_hits=0, tp2_hits=0, tp3_hits=0,
            last_updated=datetime.now(timezone.utc),
        )

@app.get("/public/dashboard/symbols", response_model=List[SymbolPerformance])
async def get_symbol_performance():
    """Get performance breakdown by symbol."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'hit_tp' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status = 'hit_sl' THEN 1 ELSE 0 END) as losses,
                    AVG(confidence) as avg_conf
                FROM signals
                GROUP BY symbol
                ORDER BY COUNT(*) DESC
            """))
            rows = result.fetchall()

        return [
            SymbolPerformance(
                symbol=row[0],
                total=row[1],
                wins=row[2],
                losses=row[3],
                win_rate=round(row[2] / (row[2] + row[3]) * 100, 1) if (row[2] + row[3]) > 0 else None,
                avg_confidence=round(row[4], 1),
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error getting symbol performance: {e}")
        return []

@app.get("/public/dashboard/daily", response_model=List[DailyPerformance])
async def get_daily_performance():
    """Get daily performance for chart."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as signals,
                    SUM(CASE WHEN status = 'hit_tp' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status = 'hit_sl' THEN 1 ELSE 0 END) as losses
                FROM signals
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at) DESC
                LIMIT 30
            """))
            rows = result.fetchall()

        return [
            DailyPerformance(
                date=str(row[0]),
                signals=row[1],
                wins=row[2],
                losses=row[3],
                win_rate=round(row[2] / (row[2] + row[3]) * 100, 1) if (row[2] + row[3]) > 0 else None,
            )
            for row in reversed(rows)  # Oldest first for chart
        ]
    except Exception as e:
        logger.error(f"Error getting daily performance: {e}")
        return []

class FeatureImportance(BaseModel):
    feature_name: str
    avg_value: float
    win_avg: float
    loss_avg: float
    win_rate: Optional[float]
    signal_count: int
    importance_score: float


class FeatureDrift(BaseModel):
    feature_name: str
    recent_avg: float
    historical_avg: float
    drift_percent: float
    is_significant: bool


@app.get("/public/dashboard/features", response_model=List[FeatureImportance])
async def get_feature_importance():
    """Get top 10 features by importance (correlation with wins)."""
    try:
        with engine.connect() as conn:
            # Check if signal_features table exists
            table_exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'signal_features'
                )
            """)).scalar()

            if not table_exists:
                return []

            # Get feature importance based on win/loss correlation
            result = conn.execute(text("""
                WITH feature_outcomes AS (
                    SELECT
                        sf.feature_name,
                        sf.feature_value,
                        CASE WHEN s.status IN ('hit_tp', 'hit_tp3') THEN 1 ELSE 0 END as is_win,
                        CASE WHEN s.status = 'hit_sl' THEN 1 ELSE 0 END as is_loss
                    FROM signal_features sf
                    JOIN signals s ON sf.signal_id = s.id AND sf.signal_created_at = s.created_at
                    WHERE s.status IN ('hit_tp', 'hit_tp3', 'hit_sl')
                    AND sf.feature_value IS NOT NULL
                ),
                feature_stats AS (
                    SELECT
                        feature_name,
                        AVG(feature_value) as avg_value,
                        AVG(CASE WHEN is_win = 1 THEN feature_value END) as win_avg,
                        AVG(CASE WHEN is_loss = 1 THEN feature_value END) as loss_avg,
                        SUM(is_win)::float / NULLIF(SUM(is_win) + SUM(is_loss), 0) * 100 as win_rate,
                        COUNT(*) as signal_count,
                        -- Importance score: absolute difference between win and loss averages
                        ABS(COALESCE(AVG(CASE WHEN is_win = 1 THEN feature_value END), 0) -
                            COALESCE(AVG(CASE WHEN is_loss = 1 THEN feature_value END), 0)) as importance_score
                    FROM feature_outcomes
                    GROUP BY feature_name
                    HAVING COUNT(*) >= 5  -- Minimum samples
                )
                SELECT * FROM feature_stats
                ORDER BY importance_score DESC
                LIMIT 10
            """))
            rows = result.fetchall()

        return [
            FeatureImportance(
                feature_name=row[0],
                avg_value=round(row[1], 4) if row[1] else 0,
                win_avg=round(row[2], 4) if row[2] else 0,
                loss_avg=round(row[3], 4) if row[3] else 0,
                win_rate=round(row[4], 1) if row[4] else None,
                signal_count=row[5],
                importance_score=round(row[6], 4) if row[6] else 0,
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return []


@app.get("/public/dashboard/features/drift", response_model=List[FeatureDrift])
async def get_feature_drift():
    """Detect feature drift - features behaving differently recently vs historically."""
    try:
        with engine.connect() as conn:
            # Check if signal_features table exists
            table_exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'signal_features'
                )
            """)).scalar()

            if not table_exists:
                return []

            # Compare recent (7 days) vs historical averages
            result = conn.execute(text("""
                WITH recent AS (
                    SELECT feature_name, AVG(feature_value) as recent_avg
                    FROM signal_features
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                    AND feature_value IS NOT NULL
                    GROUP BY feature_name
                    HAVING COUNT(*) >= 3
                ),
                historical AS (
                    SELECT feature_name, AVG(feature_value) as historical_avg
                    FROM signal_features
                    WHERE created_at < NOW() - INTERVAL '7 days'
                    AND feature_value IS NOT NULL
                    GROUP BY feature_name
                    HAVING COUNT(*) >= 10
                )
                SELECT
                    r.feature_name,
                    r.recent_avg,
                    h.historical_avg,
                    CASE
                        WHEN h.historical_avg = 0 THEN 0
                        ELSE ((r.recent_avg - h.historical_avg) / ABS(h.historical_avg)) * 100
                    END as drift_percent
                FROM recent r
                JOIN historical h ON r.feature_name = h.feature_name
                WHERE ABS(CASE
                    WHEN h.historical_avg = 0 THEN 0
                    ELSE ((r.recent_avg - h.historical_avg) / ABS(h.historical_avg)) * 100
                END) > 20  -- Only show >20% drift
                ORDER BY ABS(CASE
                    WHEN h.historical_avg = 0 THEN 0
                    ELSE ((r.recent_avg - h.historical_avg) / ABS(h.historical_avg)) * 100
                END) DESC
                LIMIT 10
            """))
            rows = result.fetchall()

        return [
            FeatureDrift(
                feature_name=row[0],
                recent_avg=round(row[1], 4) if row[1] else 0,
                historical_avg=round(row[2], 4) if row[2] else 0,
                drift_percent=round(row[3], 1) if row[3] else 0,
                is_significant=abs(row[3]) > 50 if row[3] else False,  # >50% is significant
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error getting feature drift: {e}")
        return []


@app.get("/public/dashboard/recent", response_model=List[RecentSignalDetail])
async def get_recent_signals_detail():
    """Get last 10 signals with full details for dashboard."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, symbol, direction, timeframe, confidence,
                       entry_price, stop_loss, take_profit_1, risk_reward_ratio,
                       status, created_at
                FROM signals
                ORDER BY created_at DESC
                LIMIT 10
            """))
            rows = result.fetchall()

        return [
            RecentSignalDetail(
                id=row[0],
                symbol=row[1],
                direction=row[2],
                timeframe=row[3],
                confidence=round(row[4], 1),
                entry_price=row[5],
                stop_loss=row[6],
                take_profit_1=row[7],
                risk_reward_ratio=round(row[8], 2),
                status=row[9],
                created_at=row[10],
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error getting recent signals: {e}")
        return []


class ContextWinRate(BaseModel):
    context_value: str
    total: int
    wins: int
    losses: int
    win_rate: Optional[float]
    avg_pnl: float


class PartialFillStats(BaseModel):
    total_signals: int
    tp1_rate: float
    tp2_rate: float
    tp3_rate: float
    avg_tp1_pnl: float
    avg_tp2_pnl: float
    avg_tp3_pnl: float


@app.get("/public/dashboard/context-analysis", response_model=Dict)
async def get_context_analysis():
    """Get win rate analysis by entry context (volatility, trend, volume)."""
    try:
        with engine.connect() as conn:
            # Check if trade_context table exists and has data
            table_exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'trade_context'
                )
            """)).scalar()

            if not table_exists:
                return {"volatility": [], "trend": [], "volume": []}

            # Win rate by volatility regime
            volatility_result = conn.execute(text("""
                SELECT
                    tc.volatility_regime,
                    COUNT(*) as total,
                    COUNT(CASE WHEN o.outcome = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN o.outcome = 'loss' THEN 1 END) as losses,
                    AVG(o.actual_pnl_percent) as avg_pnl
                FROM trade_context tc
                JOIN signal_outcomes o ON tc.signal_id = o.signal_id
                    AND tc.signal_created_at = o.signal_created_at
                WHERE tc.context_type = 'entry'
                    AND tc.volatility_regime IS NOT NULL
                    AND o.outcome IN ('win', 'loss')
                GROUP BY tc.volatility_regime
                ORDER BY COUNT(*) DESC
            """)).fetchall()

            volatility = [
                {
                    "context_value": row[0],
                    "total": row[1],
                    "wins": row[2],
                    "losses": row[3],
                    "win_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else None,
                    "avg_pnl": round(row[4], 2) if row[4] else 0,
                }
                for row in volatility_result
            ]

            # Win rate by trend direction
            trend_result = conn.execute(text("""
                SELECT
                    tc.trend_direction,
                    COUNT(*) as total,
                    COUNT(CASE WHEN o.outcome = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN o.outcome = 'loss' THEN 1 END) as losses,
                    AVG(o.actual_pnl_percent) as avg_pnl
                FROM trade_context tc
                JOIN signal_outcomes o ON tc.signal_id = o.signal_id
                    AND tc.signal_created_at = o.signal_created_at
                WHERE tc.context_type = 'entry'
                    AND tc.trend_direction IS NOT NULL
                    AND o.outcome IN ('win', 'loss')
                GROUP BY tc.trend_direction
                ORDER BY COUNT(*) DESC
            """)).fetchall()

            trend = [
                {
                    "context_value": row[0],
                    "total": row[1],
                    "wins": row[2],
                    "losses": row[3],
                    "win_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else None,
                    "avg_pnl": round(row[4], 2) if row[4] else 0,
                }
                for row in trend_result
            ]

            # Win rate by volume profile
            volume_result = conn.execute(text("""
                SELECT
                    tc.volume_profile,
                    COUNT(*) as total,
                    COUNT(CASE WHEN o.outcome = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN o.outcome = 'loss' THEN 1 END) as losses,
                    AVG(o.actual_pnl_percent) as avg_pnl
                FROM trade_context tc
                JOIN signal_outcomes o ON tc.signal_id = o.signal_id
                    AND tc.signal_created_at = o.signal_created_at
                WHERE tc.context_type = 'entry'
                    AND tc.volume_profile IS NOT NULL
                    AND o.outcome IN ('win', 'loss')
                GROUP BY tc.volume_profile
                ORDER BY COUNT(*) DESC
            """)).fetchall()

            volume = [
                {
                    "context_value": row[0],
                    "total": row[1],
                    "wins": row[2],
                    "losses": row[3],
                    "win_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else None,
                    "avg_pnl": round(row[4], 2) if row[4] else 0,
                }
                for row in volume_result
            ]

        return {
            "volatility": volatility,
            "trend": trend,
            "volume": volume,
        }
    except Exception as e:
        logger.error(f"Error getting context analysis: {e}")
        return {"volatility": [], "trend": [], "volume": []}


@app.get("/public/dashboard/partial-fills", response_model=PartialFillStats)
async def get_partial_fill_stats():
    """Get statistics about TP level hit rates."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN tp1_hit THEN 1 END) as tp1_count,
                    COUNT(CASE WHEN tp2_hit THEN 1 END) as tp2_count,
                    COUNT(CASE WHEN tp3_hit THEN 1 END) as tp3_count
                FROM signal_outcomes
                WHERE outcome IN ('win', 'loss', 'partial', 'expired')
            """)).fetchone()

            total = result[0] or 1  # Avoid division by zero
            tp1_count = result[1] or 0
            tp2_count = result[2] or 0
            tp3_count = result[3] or 0

            # Get average P&L for each TP level
            pnl_result = conn.execute(text("""
                SELECT
                    AVG(CASE WHEN tp1_hit AND NOT tp2_hit THEN actual_pnl_percent END) as avg_tp1_pnl,
                    AVG(CASE WHEN tp2_hit AND NOT tp3_hit THEN actual_pnl_percent END) as avg_tp2_pnl,
                    AVG(CASE WHEN tp3_hit THEN actual_pnl_percent END) as avg_tp3_pnl
                FROM signal_outcomes
                WHERE outcome IN ('win', 'loss', 'partial')
            """)).fetchone()

        return PartialFillStats(
            total_signals=total,
            tp1_rate=round(tp1_count / total * 100, 1) if total > 0 else 0,
            tp2_rate=round(tp2_count / total * 100, 1) if total > 0 else 0,
            tp3_rate=round(tp3_count / total * 100, 1) if total > 0 else 0,
            avg_tp1_pnl=round(pnl_result[0], 2) if pnl_result[0] else 0,
            avg_tp2_pnl=round(pnl_result[1], 2) if pnl_result[1] else 0,
            avg_tp3_pnl=round(pnl_result[2], 2) if pnl_result[2] else 0,
        )
    except Exception as e:
        logger.error(f"Error getting partial fill stats: {e}")
        return PartialFillStats(
            total_signals=0,
            tp1_rate=0, tp2_rate=0, tp3_rate=0,
            avg_tp1_pnl=0, avg_tp2_pnl=0, avg_tp3_pnl=0,
        )


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
