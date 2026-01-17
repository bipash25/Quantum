"""
Quantum Trading AI - Scheduler Main
"""
import asyncio
import logging
import sys

from .config import settings
from .scheduler import SignalScheduler


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
    logger.info("Quantum Trading AI - Scheduler")
    logger.info("=" * 50)
    logger.info("Schedule: 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC")

    scheduler = SignalScheduler()
    await scheduler.run()


if __name__ == "__main__":
    asyncio.run(main())
