import asyncio
from asyncio.tasks import all_tasks
from h3._cy import cells
import httpx
import os
import json
import logging
import traceback
import h3
from itertools import chain
from pathlib import Path
from shapely.geometry import shape, Polygon
from rabbitmq import RabbitMQ
from config import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def get_cell_ids_from_geojson(geojson: dict) -> list[str]:
    cell_ids = []
    for feature in geojson["features"]:
        geometry = shape(feature["geometry"])
        if not isinstance(geometry, Polygon):
            raise TypeError("geojson is not a Polygon")

        # TODO support other types
        match geometry.geom_type:
            case "Polygon":
                exterior = [[y, x] for x, y in list(geometry.exterior.coords)]
                cells = h3.polygon_to_cells(h3.LatLngPoly(exterior), resolution)
                cell_ids.extend(cells)
            case _:
                pass
    return cell_ids


async def get_all_geojson() -> list:
    async with httpx.AsyncClient(timeout=client_timeout_seconds) as client:
        res = await client.get(geojson_url)
        res.raise_for_status()
        return res.json()


async def create_snapshot(cell_ids: list[str]) -> dict:
    async with httpx.AsyncClient(timeout=client_timeout_seconds) as client:
        res = await client.post(inference_url, json=cell_ids)
        res.raise_for_status()
        return res.json()


async def main():
    rabbit_mq = RabbitMQ(broker_url, queue)
    await rabbit_mq.connect()

    all_geojson = await get_all_geojson()
    cell_ids = list(chain(*[get_cell_ids_from_geojson(x) for x in all_geojson]))
    while True:
        try:
            if len(cell_ids) == 0:
                log.info("no cells to process")
                await asyncio.sleep(1)
                continue
            log.info(f"requesting inference for {len(cell_ids)} cells")
            data = await create_snapshot(cell_ids)
            await rabbit_mq.publish(data)
        except Exception as e:
            log.error(f"failed to get inference: {traceback.format_exc()}")
        else:
            log.info(
                f"published data for {len(data.get('inferred_brightnesses', []))} cells to {queue}"
            )
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
