import logging
import asyncio

import httpx
import pika
from pika.exceptions import AMQPConnectionError

from .prediction import publish_cell_brightness
from .cells import get_h3_cells
from .config import rabbitmq_host, prediction_queue, task_sleep_interval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    """initializes process of publishing sky brightness

    n.b. with 122 res 0 cells on 2016 macbook, this will publish at a rate of 1.4m/s
    """

    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
        channel = connection.channel()
        channel.queue_declare(queue=prediction_queue)
    except AMQPConnectionError as e:
        import sys

        log.error(f"could not form amqp connection; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    except Exception as e:
        log.error(f"could not start publisher because {e}")
    else:
        try:
            h3_cell_coords = get_h3_cells()
            log.debug(f"using {len(h3_cell_coords)} resolution zero cells")

            async with httpx.AsyncClient() as client:
                while True:
                    for cell_coords in h3_cell_coords:
                        await asyncio.create_task(publish_cell_brightness(client, cell_coords, channel))
                        await asyncio.sleep(task_sleep_interval)
        except Exception as e:
            log.error(f"could not continue publishing because {e}")
            channel.close()


if __name__ == "__main__":
    asyncio.run(main())
