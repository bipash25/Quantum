"""
Quantum Trading AI - Telegram Bot Main
"""
import asyncio
import logging
import sys

from .config import settings
from .bot import TelegramBot


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
    logger.info("Quantum Trading AI - Telegram Bot")
    logger.info("=" * 50)
    logger.info(f"Channel: {settings.telegram_channel_id}")

    bot = TelegramBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
