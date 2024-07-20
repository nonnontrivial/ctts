import json
import asyncio
import logging
from dataclasses import dataclass

import psycopg
import aio_pika
import websockets

from pc.config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# FIXME should be defined elsewhere
@dataclass
class BrightnessMessage:
    uuid: str
    lat: float
    lon: float
    h3_id: str
    utc_iso: str
    utc_ns: int
    mpsas: float
    model_version: str


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


async def main():
    """create table in pg if needed and begin consuming messages from the queue,
    storing them in the predictions table"""
    initialize_db()

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
                    json_data = json.loads(m.body.decode())
                    brightness_message = BrightnessMessage(**json_data)
                    insert_brightness_message_in_db(brightness_message)

        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
