# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aio-pika",
#     "folium>=0.19.5",
#     "h3>=4.2.2",
# ]
# ///

import sqlite3
import json
import time
import asyncio
import aio_pika
import folium
import h3

snapshot_queue = "brightness.snapshot"
broker_url = "amqp://guest:guest@localhost/"

db = "brightness.db"

conn = sqlite3.connect(db)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS brightness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    brightness REAL NOT NULL
)
""")
conn.commit()


def insert_cell_brightness_records(records: list[tuple]) -> None:
    cursor.executemany(
        """INSERT INTO brightness (cell, timestamp, brightness) VALUES (?, ?, ?)""",
        records,
    )
    conn.commit()
    print(f"inserted {len(records)} records to brightness table in {db}")


def store_snapshot(snapshot: dict[str, float]) -> None:
    insert_cell_brightness_records(
        [(x, int(time.time()), y) for x, y in snapshot.items()]
    )


def generate_map(snapshot: dict[str, float]) -> None:
    def cell_brightness_to_hex_color(
        value: float, *, lower_bound=16.0, upper_bound=22.0
    ):
        normalized = (upper_bound - value) / (upper_bound - lower_bound)
        r = int(255 * (1 - normalized))
        g = int(255 * (1 - normalized))
        b = int(255 * (1 - normalized))
        return f"#{r:02x}{g:02x}{b:02x}"

    # TODO get centroid
    cell = next(iter(snapshot.keys()))
    m = folium.Map(location=list(h3.cell_to_latlng(cell)), zoom_start=10)
    min_brightness = min(snapshot.values())
    max_brightness = max(snapshot.values())
    for cell, brightness in snapshot.items():
        boundary = h3.cell_to_boundary(cell)
        folium.Polygon(
            boundary,
            color="#000000",
            fill=True,
            fill_color=cell_brightness_to_hex_color(
                brightness, lower_bound=min_brightness, upper_bound=max_brightness
            ),
            fill_opacity=0.5,
        ).add_to(m)
    m.save(f"map-{int(time.time())}.html")


async def main() -> None:
    try:
        connection = await aio_pika.connect_robust(broker_url)
        channel = await connection.channel()
        queue = await channel.declare_queue(snapshot_queue)
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    snapshot: dict | None = json.loads(message.body.decode()).get(
                        "inferred_brightnesses", None
                    )
                    if snapshot is not None:
                        store_snapshot(snapshot)
                        generate_map(snapshot)
                    else:
                        print("no snapshot found")
    except aio_pika.exceptions.AMQPError as e:
        print("failed to start consumer; are the containers running?")
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
