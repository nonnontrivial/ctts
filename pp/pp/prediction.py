import uuid
import json
from typing import Tuple
from dataclasses import asdict
from datetime import datetime
import logging

from pika.channel import Channel
import redis
import httpx
import h3

from .config import model_version, api_protocol, api_host, api_port, api_version, prediction_queue, keydb_host, \
    keydb_port
from .prediction_message import BrightnessMessage

log = logging.getLogger(__name__)

keydb = redis.Redis(host=keydb_host, port=keydb_port, db=0)

prediction_endpoint_url = f"{api_protocol}://{api_host}:{api_port}/api/{api_version}/predict"


def get_cell_id(lat, lon) -> str:
    """get the h3 cell for this lat and lon"""
    return h3.geo_to_h3(lat, lon, resolution=0)[0]


async def create_brightness_message(client: httpx.AsyncClient, h3_lat: float, h3_lon: float) -> BrightnessMessage:
    """create the object that will get published to the prediction queue."""
    res = await client.get(prediction_endpoint_url, params={"lat": h3_lat, "lon": h3_lon})
    res.raise_for_status()

    data = res.json()

    if (mpsas := data.get("sky_brightness", None)) is None:
        raise ValueError("no sky brightness reading in api response")

    utc_now = datetime.utcnow().isoformat()
    brightness_message = BrightnessMessage(
        uuid=str(uuid.uuid4()),
        lat=h3_lat,
        lon=h3_lon,
        h3_id=get_cell_id(h3_lat, h3_lon),
        utc=utc_now,
        mpsas=mpsas,
        model_version=model_version
    )
    return brightness_message


async def publish_cell_brightness(client: httpx.AsyncClient, h3_coords: Tuple[float, float], channel: Channel):
    """create and publish sky brightness at given h3 cell coords."""
    try:
        lat, lon = h3_coords

        m = await create_brightness_message(client, lat, lon)
        message_body = asdict(m)

        log.info(f"publishing {message_body} to {prediction_queue}")
        channel.basic_publish(exchange="", routing_key=prediction_queue, body=json.dumps(message_body))

        keydb.incr(m.h3_id)
        log.info(f"{m.h3_id} has had {int(keydb.get(m.h3_id))} predictions published")
    except httpx.HTTPStatusError as e:
        log.error(f"got bad status from api server {e}")
    except Exception as e:
        log.error(f"could not publish prediction at {h3_coords} because {e}")
