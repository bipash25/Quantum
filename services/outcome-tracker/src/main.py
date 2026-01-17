"""Main entry point for outcome tracker service."""
import asyncio
from .tracker import main

if __name__ == "__main__":
    asyncio.run(main())
