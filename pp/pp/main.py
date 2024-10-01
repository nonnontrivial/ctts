import logging
import asyncio

import httpx
import pika
from pika.exceptions import AMQPConnectionError

from .prediction import publish_observation_to_queue
from .cells.continent_manager import H3ContinentManager
from .config import rabbitmq_host, queue_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

continent_manager = H3ContinentManager(continent="north-america")


async def main():
    """begin publishing sky brightness predictions to the prediction queue."""
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
        channel = connection.channel()
        channel.queue_declare(queue=queue_name)
    except AMQPConnectionError as _:
        import sys

        log.error(f"could not form amqp connection; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    except Exception as e:
        log.error(f"could not start publisher because {e}")
    else:
        from collections import defaultdict
        from h3 import h3_to_geo

        try:
            na_cells = continent_manager.get_cell_covering()
            log.info(f"requesting predictions for {len(na_cells)} cells")

            cell_counts = defaultdict(int)

            async with httpx.AsyncClient() as client:
                while True:
                    for cell in na_cells:
                        cell_counts[cell] += 1

                        geo_cell = h3_to_geo(cell)
                        await asyncio.create_task(publish_observation_to_queue(client, geo_cell, channel))
                        await asyncio.sleep(0.1)
                        log.info(f"{len(cell_counts)} distinct cells have had observations published")
        except Exception as e:
            log.error(f"could not continue publishing because {e}")
            channel.close()


if __name__ == "__main__":
    asyncio.run(main())
