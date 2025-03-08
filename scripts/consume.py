# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aio-pika",
# ]
# ///

import json
import pprint
import asyncio
import aio_pika

snapshot_queue = "brightness.snapshot"
broker_url = "amqp://guest:guest@localhost/"


async def main() -> None:
    connection = await aio_pika.connect_robust(broker_url)
    channel = await connection.channel()
    queue = await channel.declare_queue(snapshot_queue)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                pprint.pprint(json.loads(message.body.decode()))


if __name__ == "__main__":
    asyncio.run(main())
