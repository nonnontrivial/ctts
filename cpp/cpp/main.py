import logging
import asyncio

import httpx
import pika

from .cells import predict_on_cell, get_res_zero_cell_coords
from .config import rabbitmq_host, prediction_queue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
    channel = connection.channel()
    channel.queue_declare(queue=prediction_queue)

    async with httpx.AsyncClient() as client:
        while True:
            cells = get_res_zero_cell_coords()
            log.info(f"found {len(cells)} resolution zero cells")
            tasks = [predict_on_cell(client, coords, channel) for coords in cells[:10]]
            await asyncio.gather(*tasks)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
