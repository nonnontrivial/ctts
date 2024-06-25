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
    res = await client.get(prediction_endpoint_url, params={"lat": lat, "lon": lon})
    res.raise_for_status()
    data = res.json()
    if (mpsas := data.get("sky_brightness", None)) is None:
        raise ValueError("no sky brightness reading in api response")

    return PredictionMessage(
        lat=lat,
        lon=lon,
        h3_id=h3.geo_to_h3(lat, lon, 0),
        utc=datetime.utcnow().isoformat(),
        mpsas=mpsas,
    )


async def predict_on_cell_coords(client: httpx.AsyncClient, coords: Tuple[float, float], channel: Channel):
    """retrieve and publish a sky brightness prediction at coords for the h3 cell"""
    import json

    try:
        lat, lon = coords

        prediction_message = await get_prediction_message_for_lat_lon(client, lat, lon)
        message_body = asdict(prediction_message)

        log.info(f"publishing prediction message {message_body} with routing key {prediction_queue}")
        channel.basic_publish(exchange="", routing_key=prediction_queue, body=json.dumps(message_body))
    except httpx.HTTPStatusError as e:
        log.error(f"got bad status from api server {e}")
    except Exception as e:
        log.error(f"could not publish prediction at {coords} because {e}")
