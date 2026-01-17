# Quantum Trading AI

AI-powered trading signal service for cryptocurrencies. Uses machine learning to generate high-confidence LONG/SHORT signals with entry, take-profit, and stop-loss levels.

## Quick Start

### Prerequisites
- AlmaLinux 9 / RHEL 9 / CentOS 9 (or similar)
- Docker & Docker Compose
- Python 3.9+

### 1. Clone and Setup

```bash
cd /root/Projects/QuantumTradingAI

# Copy environment template and update values
cp .env.example .env
nano .env  # Update TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID
```

### 2. Start Core Services

```bash
# Start database and cache
docker compose up -d timescaledb redis

# Verify services are healthy
docker compose ps
```

### 3. Run Historical Data Backfill

Before training models, you need historical data:

```bash
# Install dependencies
pip install aiohttp asyncpg

# Backfill 180 days of 1-minute data for all MVP symbols
python scripts/backfill.py --days 180 --all

# For a quick test, backfill just 7 days for BTC/ETH
python scripts/backfill.py --days 7 --symbols BTCUSDT,ETHUSDT
```

### 4. Start Data Ingestion (Real-time)

```bash
# Build and start the data ingestion service
docker compose up -d data-ingestion

# View logs
docker compose logs -f data-ingestion
```

### 5. Start All Services

```bash
docker compose up -d
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QUANTUM TRADING AI                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Binance    │───▶│    Data      │───▶│  TimescaleDB │  │
│  │   WebSocket  │    │  Ingestion   │    │  (candles)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                             │                    │          │
│                             ▼                    ▼          │
│                      ┌──────────────┐    ┌──────────────┐  │
│                      │    Redis     │    │  ML Engine   │  │
│                      │   (cache)    │    │  (training)  │  │
│                      └──────────────┘    └──────────────┘  │
│                             │                    │          │
│                             ▼                    ▼          │
│                      ┌──────────────┐    ┌──────────────┐  │
│                      │   Signal     │◀───│   Models     │  │
│                      │  Generator   │    │  (predict)   │  │
│                      └──────────────┘    └──────────────┘  │
│                             │                               │
│                             ▼                               │
│                      ┌──────────────┐                      │
│                      │  Telegram    │                      │
│                      │    Bot       │                      │
│                      └──────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
QuantumTradingAI/
├── docker-compose.yml      # Container orchestration
├── .env                    # Environment variables (secrets)
├── .env.example            # Environment template
├── services/
│   ├── data-ingestion/     # Real-time Binance data collection
│   ├── ml-engine/          # Model training and prediction
│   ├── signal-generator/   # Signal generation logic
│   ├── telegram-bot/       # Telegram delivery
│   └── api/                # REST API
├── shared/
│   ├── config/             # Shared configuration
│   ├── utils/              # Shared utilities
│   └── db/                 # Database schema
├── scripts/
│   ├── backfill.py         # Historical data backfill
│   └── test_binance.py     # Binance API test
├── data/
│   ├── models/             # Trained ML models
│   ├── backups/            # Database backups
│   └── logs/               # Application logs
└── monitoring/
    ├── prometheus/         # Metrics collection
    └── grafana/            # Dashboards
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| TimescaleDB | 5432 | Time-series database |
| Redis | 6379 | Cache and message broker |
| API | 8000 | REST API |
| Prometheus | 9090 | Metrics |
| Grafana | 3000 | Dashboards |

## Configuration

### Trading Pairs (MVP)

1. BTCUSDT (Bitcoin)
2. ETHUSDT (Ethereum)
3. SOLUSDT (Solana)
4. BNBUSDT (Binance Coin)
5. XRPUSDT (Ripple)
6. ADAUSDT (Cardano)
7. DOGEUSDT (Dogecoin)
8. MATICUSDT (Polygon)
9. DOTUSDT (Polkadot)
10. AVAXUSDT (Avalanche)
11. LINKUSDT (Chainlink)
12. UNIUSDT (Uniswap)
13. ATOMUSDT (Cosmos)
14. LTCUSDT (Litecoin)
15. NEARUSDT (Near Protocol)
16. FTMUSDT (Fantom)
17. ALGOUSDT (Algorand)
18. AAVEUSDT (Aave)
19. SANDUSDT (The Sandbox)
20. MANAUSDT (Decentraland)

### Signal Timeframes

- **4H** - 4-hour signals for swing trading
- **24H** - Daily signals for position trading

### Subscription Tiers

| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | Top 10 coins, 15-min delay |
| Starter | $19/mo | All 20 coins, real-time |
| Pro | $49/mo | + API access, backtests |
| Elite | $149/mo | + Priority support, community |

## Monitoring

Access Grafana at `http://your-server:3000`
- Default login: admin / quantum_grafana_2026

## Troubleshooting

### Database Connection Failed
```bash
# Check if container is running
docker compose ps timescaledb

# View logs
docker compose logs timescaledb

# Restart
docker compose restart timescaledb
```

### WebSocket Disconnections
Normal behavior - the service auto-reconnects with exponential backoff.

### High Memory Usage
Adjust Redis max memory in docker-compose.yml:
```yaml
command: redis-server --maxmemory 1gb
```

## License

Proprietary - All rights reserved.

## Disclaimer

This software provides trading signals for educational purposes only. Trading cryptocurrencies carries significant risk. Past performance does not guarantee future results. Always do your own research and never trade with money you cannot afford to lose.
