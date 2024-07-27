import json
import asyncio
import logging

import psycopg
import aio_pika

from pc.config import *
from pc.model import BrightnessMessage
from pc.websockets_handler import WebSocketsHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

websockets_handler = WebSocketsHandler()


def initialize_db():
    """create the predictions table if it does not exist"""
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id serial PRIMARY KEY,
                    h3_id text NOT NULL,
                    utc_iso text NOT NULL,
                    utc_ns bigint NOT NULL,
                    mpsas real NOT NULL,
                    model_version text NOT NULL
                )
            """)
            conn.commit()


def insert_brightness_message_in_db(message: BrightnessMessage):
    """insert subset of brightness message into the predictions table"""
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            log.info(f"inserting brightness message for {message.h3_id}")

            cur.execute("""
                INSERT INTO predictions (h3_id, utc_iso, utc_ns, mpsas, model_version)
                VALUES (%s, %s, %s, %s, %s)
            """, (message.h3_id, message.utc_iso, message.utc_ns, message.mpsas, message.model_version))
            conn.commit()


async def consume_from_rabbitmq():
    """create table in pg if needed and begin consuming messages from the queue,
    storing them in the predictions table"""
    try:
        amqp_connection = await aio_pika.connect_robust(f"amqp://{AMQP_USER}:{AMQP_PASSWORD}@{AMQP_HOST}")
    except Exception as e:
        import sys

        log.error(f"could not form amqp connection because {e}; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    else:
        async with amqp_connection:

            channel = await amqp_connection.channel()
            queue = await channel.declare_queue(AMQP_PREDICTION_QUEUE)

            async for m in queue:
                async with m.process():
                    # serialize the message coming over the queue and add to postgres
                    json_data = json.loads(m.body.decode())
                    message = BrightnessMessage(**json_data)

                    insert_brightness_message_in_db(message)
                    await websockets_handler.broadcast(message)

        await asyncio.Future()


async def main():
    coroutines = [websockets_handler.setup(), consume_from_rabbitmq()]
    await asyncio.gather(*coroutines)


if __name__ == "__main__":
    initialize_db()
    asyncio.run(main())
