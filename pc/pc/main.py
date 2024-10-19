import asyncio
import logging

from pc.persistence.db import initialize_db
from pc.consumer.consumer import Consumer
from pc.consumer.websocket_handler import websocket_handler
from pc.config import amqp_url, prediction_queue


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    """run the primary coroutines together"""
    consumer = Consumer(url=amqp_url, queue_name=prediction_queue, websocket_handler=websocket_handler)
    coroutines = [
        initialize_db(),
        websocket_handler.start(),
        consumer.start(),
    ]
    await asyncio.gather(*coroutines)


if __name__ == "__main__":
    asyncio.run(main())
