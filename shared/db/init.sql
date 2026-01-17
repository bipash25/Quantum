-- =============================================================================
-- Quantum Trading AI - Database Schema
-- TimescaleDB (PostgreSQL with time-series extensions)
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- CANDLES: OHLCV price data (the core time-series data)
-- =============================================================================
CREATE TABLE IF NOT EXISTS candles (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL DEFAULT 'binance',
    timeframe VARCHAR(10) NOT NULL DEFAULT '1m',
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    quote_volume DOUBLE PRECISION,
    trades INTEGER,
    PRIMARY KEY (time, symbol, exchange, timeframe)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('candles', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Enable compression for older data (saves 10-40x storage)
ALTER TABLE candles SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,exchange,timeframe'
);

-- Automatically compress chunks older than 7 days
SELECT add_compression_policy('candles', INTERVAL '7 days', if_not_exists => TRUE);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe ON candles (symbol, timeframe, time DESC);

-- =============================================================================
-- COMPUTED FEATURES: Pre-calculated technical indicators
-- =============================================================================
CREATE TABLE IF NOT EXISTS computed_features (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL DEFAULT '1m',
    feature_version VARCHAR(20) NOT NULL DEFAULT 'v1.0.0',
    -- Store features as JSONB for flexibility during development
    -- Can migrate to array/columns for performance later
    features JSONB NOT NULL,
    PRIMARY KEY (time, symbol, timeframe, feature_version)
);

SELECT create_hypertable('computed_features', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON computed_features (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_features_version ON computed_features (feature_version);

-- =============================================================================
-- SIGNALS: Generated trading signals
-- =============================================================================
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signal_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL DEFAULT 'binance',
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('LONG', 'SHORT', 'NEUTRAL')),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
    entry_price DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION NOT NULL,
    take_profit_1 DOUBLE PRECISION NOT NULL,
    take_profit_2 DOUBLE PRECISION,
    take_profit_3 DOUBLE PRECISION,
    risk_reward_ratio DOUBLE PRECISION NOT NULL,
    position_size_pct DOUBLE PRECISION DEFAULT 2.0,
    model_version VARCHAR(50) NOT NULL,
    reasoning TEXT,
    features_snapshot JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired', 'hit_tp', 'hit_sl', 'cancelled')),
    valid_until TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable('signals', 'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals (direction, created_at DESC);

-- =============================================================================
-- SIGNAL OUTCOMES: Track signal performance
-- =============================================================================
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER NOT NULL,
    signal_created_at TIMESTAMPTZ NOT NULL,
    outcome VARCHAR(20) NOT NULL CHECK (outcome IN ('win', 'loss', 'breakeven', 'expired', 'pending')),
    exit_price DOUBLE PRECISION,
    exit_time TIMESTAMPTZ,
    pnl_percent DOUBLE PRECISION,
    max_favorable_excursion DOUBLE PRECISION,  -- Best price during trade
    max_adverse_excursion DOUBLE PRECISION,    -- Worst price during trade
    duration_minutes INTEGER,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (signal_id, signal_created_at) REFERENCES signals(id, created_at)
);

CREATE INDEX IF NOT EXISTS idx_outcomes_signal ON signal_outcomes (signal_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_outcome ON signal_outcomes (outcome);

-- =============================================================================
-- USERS: Subscriber management
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    telegram_id BIGINT UNIQUE,
    telegram_username VARCHAR(100),
    email VARCHAR(255) UNIQUE,
    tier VARCHAR(20) NOT NULL DEFAULT 'free' CHECK (tier IN ('free', 'starter', 'pro', 'elite', 'enterprise')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ,
    preferences JSONB DEFAULT '{}'::jsonb,
    referral_code VARCHAR(20) UNIQUE,
    referred_by INTEGER REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_users_telegram ON users (telegram_id);
CREATE INDEX IF NOT EXISTS idx_users_tier ON users (tier);
CREATE INDEX IF NOT EXISTS idx_users_active ON users (is_active);

-- =============================================================================
-- SUBSCRIPTIONS: Payment and tier tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS subscriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    tier VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'cancelled', 'expired', 'trial')),
    stripe_subscription_id VARCHAR(100),
    stripe_customer_id VARCHAR(100),
    current_period_start TIMESTAMPTZ,
    current_period_end TIMESTAMPTZ,
    amount_cents INTEGER,
    currency VARCHAR(10) DEFAULT 'USD',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cancelled_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_subs_user ON subscriptions (user_id);
CREATE INDEX IF NOT EXISTS idx_subs_status ON subscriptions (status);
CREATE INDEX IF NOT EXISTS idx_subs_stripe ON subscriptions (stripe_subscription_id);

-- =============================================================================
-- SIGNAL DELIVERIES: Track what was sent to whom
-- =============================================================================
CREATE TABLE IF NOT EXISTS signal_deliveries (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER NOT NULL,
    signal_created_at TIMESTAMPTZ NOT NULL,
    user_id INTEGER REFERENCES users(id),
    channel VARCHAR(20) NOT NULL CHECK (channel IN ('telegram', 'discord', 'email', 'api')),
    delivered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    message_id VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'sent' CHECK (status IN ('sent', 'delivered', 'failed', 'pending')),
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    FOREIGN KEY (signal_id, signal_created_at) REFERENCES signals(id, created_at)
);

CREATE INDEX IF NOT EXISTS idx_deliveries_signal ON signal_deliveries (signal_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_user ON signal_deliveries (user_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_status ON signal_deliveries (status);

-- =============================================================================
-- MODEL VERSIONS: Track ML model versions
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL,
    asset_class VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    training_start_date TIMESTAMPTZ,
    training_end_date TIMESTAMPTZ,
    validation_metrics JSONB,
    feature_importance JSONB,
    hyperparameters JSONB,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    file_path VARCHAR(255),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_models_active ON model_versions (is_active);
CREATE INDEX IF NOT EXISTS idx_models_type ON model_versions (model_type, asset_class, timeframe);

-- =============================================================================
-- SYSTEM METRICS: For monitoring and alerting
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    labels JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (time, metric_name)
);

SELECT create_hypertable('system_metrics', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Retention policy: keep metrics for 90 days
SELECT add_retention_policy('system_metrics', INTERVAL '90 days', if_not_exists => TRUE);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for users table
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- VIEWS: Common queries as views
-- =============================================================================

-- Recent signals with outcomes
CREATE OR REPLACE VIEW v_signals_with_outcomes AS
SELECT
    s.*,
    so.outcome,
    so.exit_price,
    so.pnl_percent,
    so.duration_minutes
FROM signals s
LEFT JOIN signal_outcomes so ON s.id = so.signal_id AND s.created_at = so.signal_created_at
ORDER BY s.created_at DESC;

-- Model performance summary
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    s.model_version,
    s.symbol,
    s.timeframe,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN so.outcome = 'win' THEN 1 END) as wins,
    COUNT(CASE WHEN so.outcome = 'loss' THEN 1 END) as losses,
    ROUND(
        COUNT(CASE WHEN so.outcome = 'win' THEN 1 END)::NUMERIC /
        NULLIF(COUNT(CASE WHEN so.outcome IN ('win', 'loss') THEN 1 END), 0) * 100,
        2
    ) as win_rate,
    ROUND(AVG(CASE WHEN so.outcome IN ('win', 'loss') THEN so.pnl_percent END)::NUMERIC, 2) as avg_pnl
FROM signals s
LEFT JOIN signal_outcomes so ON s.id = so.signal_id AND s.created_at = so.signal_created_at
WHERE s.created_at > NOW() - INTERVAL '30 days'
GROUP BY s.model_version, s.symbol, s.timeframe
ORDER BY s.model_version, win_rate DESC;

-- User tier distribution
CREATE OR REPLACE VIEW v_user_stats AS
SELECT
    tier,
    COUNT(*) as user_count,
    COUNT(CASE WHEN is_active THEN 1 END) as active_count,
    COUNT(CASE WHEN last_active_at > NOW() - INTERVAL '7 days' THEN 1 END) as weekly_active
FROM users
GROUP BY tier
ORDER BY
    CASE tier
        WHEN 'enterprise' THEN 1
        WHEN 'elite' THEN 2
        WHEN 'pro' THEN 3
        WHEN 'starter' THEN 4
        WHEN 'free' THEN 5
    END;

-- =============================================================================
-- INSERT DEFAULT DATA
-- =============================================================================

-- Insert a system user for internal operations
INSERT INTO users (telegram_id, telegram_username, email, tier)
VALUES (0, 'system', 'system@quantum-trading.ai', 'enterprise')
ON CONFLICT (telegram_id) DO NOTHING;

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================
-- These will be executed with the quantum user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO quantum;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO quantum;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO quantum;
