import uuid
import json
import logging
from typing import Tuple
from datetime import datetime

from pika.channel import Channel
import redis
import httpx

from .config import model_version, api_protocol, api_host, api_port, api_version, queue_name, keydb_host, \
    keydb_port
from .models.models import BrightnessObservation
from .cells.continent_manager import H3ContinentManager

log = logging.getLogger(__name__)

keydb = redis.Redis(host=keydb_host, port=keydb_port, db=0)


async def create_brightness_observation(client: httpx.AsyncClient, h3_lat: float,
                                        h3_lon: float) -> BrightnessObservation:
    """create the object that will get published to the prediction queue."""
    prediction_endpoint_url = f"{api_protocol}://{api_host}:{api_port}/api/{api_version}/predict"

    res = await client.get(prediction_endpoint_url, params={"lat": h3_lat, "lon": h3_lon})
    res.raise_for_status()

    data = res.json()

    if (magnitudes := data.get("mpsas", None)) is None:
        raise ValueError("no sky brightness mpsas reading in api response")

    utc_now = datetime.utcnow()
    brightness_message = BrightnessObservation(
        uuid=str(uuid.uuid4()),
        lat=h3_lat,
        lon=h3_lon,
        h3_id=H3ContinentManager.get_cell_id(h3_lat, h3_lon),
        utc_iso=utc_now.isoformat(),
        mpsas=magnitudes,
        model_version=model_version
    )
    return brightness_message


async def publish_observation_to_queue(client: httpx.AsyncClient, h3_coords: Tuple[float, float], channel: Channel):
    """request and publish sky brightness at given h3 cell coords."""
    lat, lon = h3_coords

    try:
        observation = await create_brightness_observation(client, lat, lon)
        log.info(f"publishing brightness observation for cell {observation.h3_id}")
        channel.basic_publish(exchange="", routing_key=queue_name, body=json.dumps(observation.model_dump()))
    except httpx.HTTPStatusError as e:
        log.error(f"got bad status from api server {e}")
    except Exception as e:
        import traceback

        log.error("failed to publish prediction!")
        log.error(traceback.format_exc())
    else:
        keydb.incr(observation.h3_id)
        log.info(f"cell {observation.h3_id} has had {int(keydb.get(observation.h3_id))} predictions published")
