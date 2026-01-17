"""
Quantum Trading AI - Signal Generator Main
"""
import asyncio
import logging
import sys

from .config import settings
from .generator import SignalGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("Quantum Trading AI - Signal Generator")
    logger.info("=" * 50)
    logger.info(f"Timeframe: {settings.signal_timeframe}")
    logger.info(f"Min confidence: {settings.min_confidence}")
    logger.info(f"Generation interval: {settings.generation_interval}s")
    logger.info(f"Model directory: {settings.model_dir}")

    generator = SignalGenerator()

    # Generate signals once on startup
    signals = generator.run_once()
    logger.info(f"Initial generation: {len(signals)} signals")

    # Run in loop
    await generator.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
