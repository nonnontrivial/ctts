from typing import Tuple
from dataclasses import asdict
from datetime import datetime
import asyncio
import logging

from pika.channel import Channel
import httpx

from .config import api_protocol, api_host, api_port, api_version, sleep_interval
from .message import PredictionMessage

log = logging.getLogger(__name__)

prediction_endpoint_url = f"{api_protocol}://{api_host}:{api_port}/api/{api_version}/predict"


async def get_prediction_message_for_lat_lon(client: httpx.AsyncClient, lat: float, lon: float) -> PredictionMessage:
    res = await client.get(prediction_endpoint_url, params={"lat": lat, "lon": lon})
    res.raise_for_status()

    data = res.json()
    return PredictionMessage(
        lat=lat,
        lon=lon,
        time_of=datetime.utcnow().isoformat(),
        sky_brightness_mpsas=data["sky_brightness"],
    )


async def predict_on_cell(client: httpx.AsyncClient, coords: Tuple[float, float], channel: Channel):
    import json

    try:
        lat, lon = coords
        prediction_message = await get_prediction_message_for_lat_lon(client, lat, lon)
        message_body = asdict(prediction_message)

        log.info(f"publishing prediction message {message_body}")

        # FIXME exchange, routing key
        channel.basic_publish(exchange="", routing_key="test", body=json.dumps(message_body))
    except httpx.HTTPStatusError as e:
        log.error(f"got bad status from api server {e}")
    except Exception as e:
        log.error(f"could not publish prediction at {coords} because {e}")
    finally:
        # await asyncio.sleep(sleep_interval)
        pass
