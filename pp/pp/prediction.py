from typing import Tuple
from dataclasses import asdict
from datetime import datetime
import logging

from pika.channel import Channel
import httpx
import h3

from .config import api_protocol, api_host, api_port, api_version, prediction_queue
from .message import PredictionMessage

log = logging.getLogger(__name__)

prediction_endpoint_url = f"{api_protocol}://{api_host}:{api_port}/api/{api_version}/predict"


async def get_prediction_message_for_lat_lon(client: httpx.AsyncClient, lat: float, lon: float) -> PredictionMessage:
    """create the object that will get published to rabbitmq"""
    res = await client.get(prediction_endpoint_url, params={"lat": lat, "lon": lon})
    res.raise_for_status()

    data = res.json()
    if (mpsas := data.get("sky_brightness", None)) is None:
        raise ValueError("no sky brightness reading in api response")

    message = PredictionMessage(
        lat=lat,
        lon=lon,
        h3_id=h3.geo_to_h3(lat, lon, 0),
        utc=datetime.utcnow().isoformat(),
        mpsas=mpsas,
    )
    return message


# message_store = {}


async def predict_on_cell_coords(client: httpx.AsyncClient, h3_coords: Tuple[float, float], channel: Channel):
    """retrieve and publish a sky brightness prediction at coords for the h3 cell"""
    import json

    try:
        lat, lon = h3_coords

        m = await get_prediction_message_for_lat_lon(client, lat, lon)
        message_body = asdict(m)

        log.info(f"publishing {message_body} to {prediction_queue}")
        channel.basic_publish(exchange="", routing_key=prediction_queue, body=json.dumps(message_body))

        # keep track of how many messages are published for each cell
        # message_store[m.h3_id] = message_store.get(m.h3_id, 0) + 1
        # with open("data.json", "w") as f:
        #     json.dump(message_store, f, indent=4)

    except httpx.HTTPStatusError as e:
        log.error(f"got bad status from api server {e}")
    except Exception as e:
        log.error(f"could not publish prediction at {h3_coords} because {e}")
