import asyncio
import logging

from pc.persistence.db import initialize_db
from pc.consumer.consumer import Consumer
from pc.config import amqp_url, prediction_queue, cycle_queue


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    """run the primary coroutines together"""
    consumer = Consumer(url=amqp_url, prediction_queue=prediction_queue, cycle_queue=cycle_queue)
    coroutines = [
        initialize_db(),
        consumer.start()
    ]
    await asyncio.gather(*coroutines)


if __name__ == "__main__":
    asyncio.run(main())
