import asyncio
import logging

from pc.persistence.db import create_pool, create_brightness_table
from pc.consumer.consumer import Consumer
from pc.config import amqp_url, prediction_queue, cycle_queue


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    pool = await create_pool()
    if pool is None:
        raise ValueError("no connection pool!")
    await create_brightness_table(pool)
    consumer = Consumer(
        url=amqp_url,
        prediction_queue=prediction_queue,
        cycle_queue=cycle_queue,
        connection_pool=pool,
        on_cycle_completion=lambda brightness_observation: log.info(brightness_observation.model_dump())
    )
    await consumer.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.error(f"failed to run: {e}")
