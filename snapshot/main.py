import asyncio
import httpx
import os
import json
import logging
import h3
from shapely.geometry import shape, Polygon
from pathlib import Path
from rabbitmq import RabbitMQ

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

resolution = int(os.environ.get("RESOLUTION", 0))
api_host = os.environ.get("API_HOST", "localhost")
api_port = int(os.environ.get("API_PORT", 8000))

rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")
rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
queue = os.getenv("QUEUE", "brightness.snapshot")

broker_url = f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}"
inference_url = f"http://{api_host}:{api_port}/infer"


def get_cell_ids(geojson: dict) -> list[str]:
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


async def main():
    geojson_path = Path("./data.geojson")
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found at {geojson_path}")

    geojson = json.loads(geojson_path.read_text())
    cell_ids = get_cell_ids(geojson)
    rabbit_mq = RabbitMQ(broker_url, queue)
    await rabbit_mq.connect()
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(inference_url, json=cell_ids)
                response.raise_for_status()
                data = response.json()
                await rabbit_mq.publish(data)
                log.info(
                    f"published data for {len(data['inferred_brightnesses'])} cells to {queue}"
                )
        except Exception as e:
            import traceback

            log.error(f"failed to get inference: {traceback.format_exc()}")
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
