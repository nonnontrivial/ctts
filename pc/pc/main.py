import asyncio
import logging
import typing

from pc.persistence.db import create_pg_connection_pool, setup_table
from pc.persistence.models import BrightnessObservation
from pc.consumer.consumer import Consumer
from pc.config import amqp_url, prediction_queue, cycle_queue


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def on_cycle_completion(max_observation: BrightnessObservation):
    # TODO communicate this result to cycle recipients
    log.info(f"cycle completed with max observation {max_observation.model_dump()}")


async def consume_brightness():
    pool = await create_pg_connection_pool()
    if pool is None:
        raise ValueError("no connection pool!")
    await setup_table(pool)

    consumer = Consumer(
        url=amqp_url,
        prediction_queue=prediction_queue,
        cycle_queue=cycle_queue,
        connection_pool=pool,
        on_cycle_completion=on_cycle_completion
    )
    await consumer.connect()
    await consumer.consume_from_queues()


if __name__ == "__main__":
    try:
        asyncio.run(consume_brightness())
    except Exception as e:
        log.error(f"failed to consume brightness: {e}")
