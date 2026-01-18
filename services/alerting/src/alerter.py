#!/usr/bin/env python3
"""
Quantum Trading AI - Comprehensive Alerting System
===================================================
Monitors system health and sends alerts via Telegram.

Alert Levels:
- CRITICAL: Immediate notification (service down, data loss)
- WARNING: Important issues needing attention
- INFO: Dashboard logging only (no notification)
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import aiohttp
import docker
import psutil
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_HOST = os.environ.get("DB_HOST", "timescaledb")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_USER = os.environ.get("DB_USER", "quantum")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "quantum_trading")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_CHAT_ID = os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "")

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
ALERT_CHECK_INTERVAL = int(os.environ.get("ALERT_CHECK_INTERVAL", "60"))

# Required Docker services
REQUIRED_SERVICES = [
    "quantum_timescaledb",
    "quantum_redis",
    "quantum_data_ingestion",
    "quantum_scheduler",
    "quantum_telegram_bot",
    "quantum_outcome_tracker",
    "quantum_api",
    "quantum_web",
]


class AlertLevel(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Alert:
    level: AlertLevel
    message: str
    service: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


# =============================================================================
# ALERT STATE TRACKING (prevent spam)
# =============================================================================

class AlertState:
    """Track alert state to prevent duplicate notifications."""

    def __init__(self):
        self.active_alerts: Dict[str, datetime] = {}
        self.service_down_since: Dict[str, datetime] = {}
        self.cooldown_minutes = {
            AlertLevel.CRITICAL: 15,  # Re-alert every 15 min if still critical
            AlertLevel.WARNING: 60,   # Re-alert every hour for warnings
            AlertLevel.INFO: 0,       # No notifications for INFO
        }

    def should_send(self, alert_key: str, level: AlertLevel) -> bool:
        """Check if we should send this alert (not in cooldown)."""
        if level == AlertLevel.INFO:
            return False  # Never send INFO via Telegram

        now = datetime.now(timezone.utc)
        last_sent = self.active_alerts.get(alert_key)

        if last_sent is None:
            return True

        cooldown = timedelta(minutes=self.cooldown_minutes[level])
        return (now - last_sent) > cooldown

    def mark_sent(self, alert_key: str):
        """Mark alert as sent."""
        self.active_alerts[alert_key] = datetime.now(timezone.utc)

    def clear(self, alert_key: str):
        """Clear alert when resolved."""
        self.active_alerts.pop(alert_key, None)
        self.service_down_since.pop(alert_key, None)


alert_state = AlertState()


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def get_db_engine():
    """Create database engine."""
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def log_alert_to_db(engine, alert: Alert):
    """Log alert to database."""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alerts (level, message, service, metric_value, threshold)
                VALUES (:level, :message, :service, :metric_value, :threshold)
            """), {
                "level": alert.level.value,
                "message": alert.message,
                "service": alert.service,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
            })
    except Exception as e:
        logger.error(f"Failed to log alert to DB: {e}")


# =============================================================================
# TELEGRAM NOTIFICATIONS
# =============================================================================

async def send_telegram_alert(alert: Alert):
    """Send alert via Telegram to admin chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_CHAT_ID:
        logger.warning("Telegram credentials not configured")
        return

    # Format message with emoji based on level
    emoji = {
        AlertLevel.CRITICAL: "\U0001F6A8",  # rotating light
        AlertLevel.WARNING: "\u26A0\uFE0F",  # warning
        AlertLevel.INFO: "\u2139\uFE0F",     # info
    }

    message = f"{emoji[alert.level]} *{alert.level.value}*\n\n"
    message += f"{alert.message}\n"

    if alert.service:
        message += f"\n*Service:* `{alert.service}`"
    if alert.metric_value is not None:
        message += f"\n*Value:* `{alert.metric_value:.2f}`"
    if alert.threshold is not None:
        message += f"\n*Threshold:* `{alert.threshold:.2f}`"

    message += f"\n\n_Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_ADMIN_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram send failed: {await resp.text()}")
                else:
                    logger.info(f"Telegram alert sent: {alert.level.value}")
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


# =============================================================================
# HEALTH CHECKS
# =============================================================================

async def check_docker_services() -> List[Alert]:
    """Check if all required Docker services are running."""
    alerts = []
    now = datetime.now(timezone.utc)

    try:
        client = docker.from_env()
        running_containers = {c.name for c in client.containers.list()}

        for service in REQUIRED_SERVICES:
            if service not in running_containers:
                # Track when service went down
                if service not in alert_state.service_down_since:
                    alert_state.service_down_since[service] = now

                down_duration = now - alert_state.service_down_since[service]

                if down_duration > timedelta(minutes=5):
                    alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        message=f"Docker service `{service}` has been DOWN for {down_duration.seconds // 60} minutes!",
                        service=service,
                        metric_value=down_duration.seconds / 60,
                        threshold=5.0,
                    ))
                elif down_duration > timedelta(minutes=1):
                    alerts.append(Alert(
                        level=AlertLevel.WARNING,
                        message=f"Docker service `{service}` is DOWN (checking...)",
                        service=service,
                    ))
            else:
                # Service is up, clear any tracked downtime
                alert_state.service_down_since.pop(service, None)
                alert_state.clear(f"docker_{service}")

    except Exception as e:
        alerts.append(Alert(
            level=AlertLevel.CRITICAL,
            message=f"Cannot connect to Docker daemon: {e}",
            service="docker",
        ))

    return alerts


async def check_database_connection(engine) -> List[Alert]:
    """Check database connectivity."""
    alerts = []

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        alert_state.clear("db_connection")
    except OperationalError as e:
        alerts.append(Alert(
            level=AlertLevel.CRITICAL,
            message=f"Database connection LOST: {str(e)[:100]}",
            service="timescaledb",
        ))
    except Exception as e:
        alerts.append(Alert(
            level=AlertLevel.CRITICAL,
            message=f"Database error: {str(e)[:100]}",
            service="timescaledb",
        ))

    return alerts


async def check_data_ingestion(engine) -> List[Alert]:
    """Check if new candles are being ingested."""
    alerts = []

    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT MAX(time) as last_candle,
                       NOW() - MAX(time) as age
                FROM candles
                WHERE timeframe = '1m'
            """)).fetchone()

            if result and result[1]:
                age_minutes = result[1].total_seconds() / 60

                if age_minutes > 10:
                    alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        message=f"Data ingestion STOPPED! No new candles in {age_minutes:.0f} minutes",
                        service="data_ingestion",
                        metric_value=age_minutes,
                        threshold=10.0,
                    ))
                elif age_minutes > 5:
                    alerts.append(Alert(
                        level=AlertLevel.WARNING,
                        message=f"Data ingestion delayed: {age_minutes:.0f} minutes since last candle",
                        service="data_ingestion",
                        metric_value=age_minutes,
                        threshold=5.0,
                    ))
                else:
                    alert_state.clear("data_ingestion")

    except Exception as e:
        logger.error(f"Failed to check data ingestion: {e}")

    return alerts


async def check_model_performance(engine) -> List[Alert]:
    """Check model win rates."""
    alerts = []

    try:
        with engine.connect() as conn:
            # Get recent win rate (last 7 days)
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome IN ('tp_hit', 'partial_tp') THEN 1 ELSE 0 END) as wins,
                    timeframe
                FROM signal_outcomes
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY timeframe
                HAVING COUNT(*) >= 20
            """)).fetchall()

            for row in result:
                total, wins, timeframe = row
                win_rate = wins / total if total > 0 else 0

                if win_rate < 0.40:
                    alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        message=f"Model win rate CRITICAL for {timeframe}: {win_rate:.1%} ({wins}/{total} signals)",
                        service=f"model_{timeframe}",
                        metric_value=win_rate * 100,
                        threshold=40.0,
                    ))
                elif win_rate < 0.50:
                    alerts.append(Alert(
                        level=AlertLevel.WARNING,
                        message=f"Model win rate LOW for {timeframe}: {win_rate:.1%}",
                        service=f"model_{timeframe}",
                        metric_value=win_rate * 100,
                        threshold=50.0,
                    ))
                else:
                    alert_state.clear(f"model_{timeframe}")

            # Check for win rate drop vs baseline
            baseline_result = conn.execute(text("""
                WITH recent AS (
                    SELECT
                        timeframe,
                        AVG(CASE WHEN outcome IN ('tp_hit', 'partial_tp') THEN 1.0 ELSE 0.0 END) as recent_rate
                    FROM signal_outcomes
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY timeframe
                ),
                baseline AS (
                    SELECT
                        timeframe,
                        AVG(CASE WHEN outcome IN ('tp_hit', 'partial_tp') THEN 1.0 ELSE 0.0 END) as baseline_rate
                    FROM signal_outcomes
                    WHERE created_at BETWEEN NOW() - INTERVAL '7 days' AND NOW() - INTERVAL '24 hours'
                    GROUP BY timeframe
                )
                SELECT r.timeframe, r.recent_rate, b.baseline_rate
                FROM recent r
                JOIN baseline b ON r.timeframe = b.timeframe
                WHERE b.baseline_rate > 0
            """)).fetchall()

            for row in baseline_result:
                timeframe, recent_rate, baseline_rate = row
                drop_pct = ((baseline_rate - recent_rate) / baseline_rate) * 100

                if drop_pct > 10:
                    alerts.append(Alert(
                        level=AlertLevel.WARNING,
                        message=f"Win rate DROP for {timeframe}: {drop_pct:.1f}% below 7-day baseline",
                        service=f"model_{timeframe}",
                        metric_value=drop_pct,
                        threshold=10.0,
                    ))

    except Exception as e:
        logger.error(f"Failed to check model performance: {e}")

    return alerts


async def check_system_resources() -> List[Alert]:
    """Check disk and memory usage."""
    alerts = []

    # Check disk usage
    try:
        disk = psutil.disk_usage("/")
        disk_pct = disk.percent

        if disk_pct > 90:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Disk usage CRITICAL: {disk_pct:.1f}%",
                service="system",
                metric_value=disk_pct,
                threshold=90.0,
            ))
        elif disk_pct > 80:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Disk usage HIGH: {disk_pct:.1f}%",
                service="system",
                metric_value=disk_pct,
                threshold=80.0,
            ))
        else:
            alert_state.clear("disk_usage")

    except Exception as e:
        logger.error(f"Failed to check disk usage: {e}")

    # Check memory usage
    try:
        memory = psutil.virtual_memory()
        mem_pct = memory.percent

        if mem_pct > 90:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Memory usage CRITICAL: {mem_pct:.1f}%",
                service="system",
                metric_value=mem_pct,
                threshold=90.0,
            ))
        elif mem_pct > 80:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Memory usage HIGH: {mem_pct:.1f}%",
                service="system",
                metric_value=mem_pct,
                threshold=80.0,
            ))
        else:
            alert_state.clear("memory_usage")

    except Exception as e:
        logger.error(f"Failed to check memory usage: {e}")

    return alerts


async def check_error_rates(engine) -> List[Alert]:
    """Check for high error rates in recent operations."""
    alerts = []

    try:
        with engine.connect() as conn:
            # Check recent alert counts (errors logged)
            result = conn.execute(text("""
                SELECT
                    COUNT(*) FILTER (WHERE level = 'CRITICAL') as critical_count,
                    COUNT(*) FILTER (WHERE level = 'WARNING') as warning_count,
                    COUNT(*) as total
                FROM alerts
                WHERE created_at > NOW() - INTERVAL '1 hour'
                  AND acknowledged = false
            """)).fetchone()

            if result:
                critical, warning, total = result
                if critical and critical > 10:
                    alerts.append(Alert(
                        level=AlertLevel.WARNING,
                        message=f"High error rate: {critical} critical alerts in last hour",
                        service="alerting",
                        metric_value=float(critical),
                        threshold=10.0,
                    ))

    except Exception as e:
        logger.error(f"Failed to check error rates: {e}")

    return alerts


async def check_outcome_tracker(engine) -> List[Alert]:
    """Check if outcome tracker is processing signals."""
    alerts = []

    try:
        with engine.connect() as conn:
            # Check for pending outcomes not processed
            result = conn.execute(text("""
                SELECT COUNT(*) as pending
                FROM signals s
                LEFT JOIN signal_outcomes o ON s.id = o.signal_id
                WHERE o.id IS NULL
                  AND s.created_at < NOW() - INTERVAL '2 hours'
                  AND s.created_at > NOW() - INTERVAL '48 hours'
            """)).fetchone()

            if result and result[0] > 20:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    message=f"Outcome tracker backlog: {result[0]} signals pending >2 hours",
                    service="outcome_tracker",
                    metric_value=float(result[0]),
                    threshold=20.0,
                ))

    except Exception as e:
        logger.error(f"Failed to check outcome tracker: {e}")

    return alerts


# =============================================================================
# MAIN ALERTING LOOP
# =============================================================================

async def run_health_checks(engine) -> List[Alert]:
    """Run all health checks and collect alerts."""
    all_alerts = []

    # Run all checks concurrently
    check_results = await asyncio.gather(
        check_docker_services(),
        check_database_connection(engine),
        check_data_ingestion(engine),
        check_model_performance(engine),
        check_system_resources(),
        check_error_rates(engine),
        check_outcome_tracker(engine),
        return_exceptions=True,
    )

    for result in check_results:
        if isinstance(result, Exception):
            logger.error(f"Health check failed: {result}")
            all_alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Health check error: {str(result)[:100]}",
                service="alerting",
            ))
        elif isinstance(result, list):
            all_alerts.extend(result)

    return all_alerts


async def process_alerts(engine, alerts: List[Alert]):
    """Process alerts: log to DB and send notifications."""
    for alert in alerts:
        # Generate unique key for deduplication
        alert_key = f"{alert.level.value}_{alert.service}_{alert.message[:50]}"

        # Always log to database
        log_alert_to_db(engine, alert)

        # Send Telegram notification if not in cooldown
        if alert_state.should_send(alert_key, alert.level):
            await send_telegram_alert(alert)
            alert_state.mark_sent(alert_key)
            logger.info(f"Alert sent: [{alert.level.value}] {alert.message}")
        else:
            logger.debug(f"Alert in cooldown: [{alert.level.value}] {alert.message}")


async def main():
    """Main alerting loop."""
    logger.info("=" * 60)
    logger.info("Quantum Trading AI - Alerting Service Starting")
    logger.info("=" * 60)
    logger.info(f"Check interval: {ALERT_CHECK_INTERVAL} seconds")
    logger.info(f"Admin chat ID: {TELEGRAM_ADMIN_CHAT_ID[:5]}..." if TELEGRAM_ADMIN_CHAT_ID else "Admin chat: NOT CONFIGURED")

    # Create database engine
    engine = get_db_engine()

    # Test database connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        # Send critical alert if possible
        await send_telegram_alert(Alert(
            level=AlertLevel.CRITICAL,
            message=f"Alerting service cannot connect to database: {e}",
            service="alerting",
        ))

    # Send startup notification
    await send_telegram_alert(Alert(
        level=AlertLevel.INFO,
        message="Alerting service started and monitoring system health",
        service="alerting",
    ))

    # Main loop
    while True:
        try:
            logger.debug("Running health checks...")
            alerts = await run_health_checks(engine)

            if alerts:
                critical_count = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
                warning_count = sum(1 for a in alerts if a.level == AlertLevel.WARNING)
                logger.info(f"Health check complete: {critical_count} critical, {warning_count} warnings")
                await process_alerts(engine, alerts)
            else:
                logger.debug("Health check complete: All systems nominal")

        except Exception as e:
            logger.error(f"Error in alerting loop: {e}")
            await send_telegram_alert(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Alerting service error: {str(e)[:200]}",
                service="alerting",
            ))

        await asyncio.sleep(ALERT_CHECK_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
