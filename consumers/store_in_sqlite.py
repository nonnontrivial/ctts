# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aio-pika",
# ]
# ///

import sqlite3
import json
import time
import asyncio
import aio_pika

snapshot_queue = "brightness.snapshot"
broker_url = "amqp://guest:guest@localhost/"

db = "brightness.db"
conn = sqlite3.connect(db)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS brightness (
    cell TEXT PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    brightness REAL NOT NULL
)
""")
conn.commit()


def insert_records(records: list[tuple]) -> None:
    cursor.executemany(
        """
    INSERT OR REPLACE INTO brightness (cell, timestamp, brightness)
    VALUES (?, ?, ?)
    """,
        records,
    )
    print(f"inserted {len(records)} records to {db}")
    conn.commit()


async def main() -> None:
    try:
        connection = await aio_pika.connect_robust(broker_url)
        channel = await connection.channel()
        queue = await channel.declare_queue(snapshot_queue)
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    data = json.loads(message.body.decode()).get(
                        "inferred_brightnesses", None
                    )
                    if data is not None:
                        insert_records(
                            [(x, int(time.time()), y) for x, y in data.items()]
                        )
    except KeyboardInterrupt:
        pass
    except aio_pika.exceptions.AMQPError as e:
        print("failed to start consumer; are the main containers running?")
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
