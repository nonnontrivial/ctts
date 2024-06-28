from typing import Tuple
from dataclasses import asdict
from datetime import datetime
import logging

from pika.channel import Channel
import redis
import httpx
import h3

from .config import api_protocol, api_host, api_port, api_version, prediction_queue, keydb_host, keydb_port
from .message import PredictionMessage

log = logging.getLogger(__name__)
keydb = redis.Redis(host=keydb_host, port=keydb_port, db=0)

prediction_endpoint_url = f"{api_protocol}://{api_host}:{api_port}/api/{api_version}/predict"


async def get_prediction_message(client: httpx.AsyncClient, h3_lat: float, h3_lon: float) -> PredictionMessage:
    """create the object that will get published to rabbitmq."""
    res = await client.get(prediction_endpoint_url, params={"lat": h3_lat, "lon": h3_lon})
    res.raise_for_status()

    data = res.json()
    if (mpsas := data.get("sky_brightness", None)) is None:
        raise ValueError("no sky brightness reading in api response")

    message = PredictionMessage(
        lat=h3_lat,
        lon=h3_lon,
        utc=datetime.utcnow().isoformat(),
        mpsas=mpsas,
        h3_id=h3.geo_to_h3(h3_lat, h3_lon, resolution=0),
    )
    return message


async def publish_cell_prediction(client: httpx.AsyncClient, h3_coords: Tuple[float, float], channel: Channel):
    """retrieve and publish a sky brightness prediction at h3 cell coords."""
    import json

    try:
        lat, lon = h3_coords
        m = await get_prediction_message(client, lat, lon)
        message_body = asdict(m)

        log.info(f"publishing {message_body} to {prediction_queue}")
        channel.basic_publish(exchange="", routing_key=prediction_queue, body=json.dumps(message_body))

        keydb.incr(m.h3_id)
        num_predictions_published = int(keydb.get(m.h3_id))
        log.info(f"{m.h3_id} has had {num_predictions_published} predictions published")

    except httpx.HTTPStatusError as e:
        log.error(f"got bad status from api server {e}")
    except Exception as e:
        log.error(f"could not publish prediction at {h3_coords} because {e}")
