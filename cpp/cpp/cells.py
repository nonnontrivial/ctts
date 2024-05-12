from typing import Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from pika.channel import Channel
import h3
import httpx

from .config import api_protocol, api_host, api_port, api_version

log = logging.getLogger(__name__)


@dataclass
class PredictionMessage:
    time_of: str
    lat: float
    lon: float
    sky_brightness_mpsas: float


async def predict_on_cell(client: httpx.AsyncClient, coords: Tuple[float, float], channel: Channel):
    """generate the prediction for the cell coordinates; publishing to rabbitmq once available"""
    import json
    import asyncio

    api_url = f"{api_protocol}://{api_host}:{api_port}/api/{api_version}/prediction"

    lat, lon = coords
    res = await client.get(api_url, params={"lat": lat, "lon": lon})
    # res.raise_for_status()
    data = res.json()
    message_body = asdict(PredictionMessage(
        lat=lat,
        lon=lon,
        time_of=datetime.utcnow().isoformat(),
        sky_brightness_mpsas=data["sky_brightness"],
    ))
    log.info(f"publishing prediction message {message_body}")
    # FIXME routing key
    channel.basic_publish(exchange="", routing_key="hello", body=json.dumps(message_body))
    await asyncio.sleep(1)


def get_res_zero_cell_coords() -> List[Tuple[float, float]]:
    """get coordinates of all resolution zero cells"""
    resolution_zero_cells = h3.get_res0_indexes()
    return [h3.h3_to_geo(c) for c in resolution_zero_cells]
