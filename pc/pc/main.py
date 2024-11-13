import asyncio
import logging
import typing

from pc.persistence.db import create_connection_pool, create_brightness_table
from pc.persistence.models import BrightnessObservation
from pc.consumer.consumer import Consumer
from pc.config import amqp_url, prediction_queue, cycle_queue


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def on_cycle_completion(brightness_observation: BrightnessObservation):
    log.info(f"cycle completed with {brightness_observation.model_dump()}")


async def main():
    pool = await create_connection_pool()
    if pool is None:
        raise ValueError("no connection pool!")

    await create_brightness_table(pool)
    consumer = Consumer(
        url=amqp_url,
        prediction_queue=prediction_queue,
        cycle_queue=cycle_queue,
        connection_pool=pool,
        on_cycle_completion=on_cycle_completion
    )
    await consumer.connect()
    await consumer.consume()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.error(f"failed to run: {e}")
