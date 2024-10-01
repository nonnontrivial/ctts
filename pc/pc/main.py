import asyncio
import logging

from pc.persistence.db import initialize_db
from pc.rabbitmq import consume_brightness_observations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    """run the primary coroutines together"""
    coroutines = [initialize_db(), consume_brightness_observations()]
    await asyncio.gather(*coroutines)


if __name__ == "__main__":
    asyncio.run(main())
