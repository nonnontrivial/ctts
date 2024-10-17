import typing as t

import requests

from .. import config

from ..observer_site import ObserverSite
from ..utils import get_astro_time_hour

model = config["meteo"]["model"]

class OpenMeteoClient:
    def __init__(self, site: ObserverSite) -> None:
        self.site = site

        protocol=config["meteo"]["protocol"]
        host = config["meteo"]["host"]
        port = config["meteo"]["port"]
        self.url_base = f"{protocol}://{host}:{port}"

    def get_forecast(self) -> t.Tuple[int, float]:
        lat, lon = self.site.latitude.value, self.site.longitude.value

        hourly_params = {"temperature_2m", "cloud_cover"}
        params = {
            "latitude": lat,
            "longitude": lon,
            "models": model,
            "hourly": ",".join(hourly_params)
        }
        r = requests.get(f"{self.url_base}/v1/forecast", params=params)
        r.raise_for_status()

        res_json = r.json()
        elevation = float(res_json.get("elevation", 0.))

        idx = self.get_hourly_index_of_site_time()
        cloud_cover = res_json["hourly"]["cloud_cover"][idx]
        cloud_cover = self.get_cloud_cover_as_oktas(cloud_cover)

        return cloud_cover, elevation

    def get_hourly_index_of_site_time(self) -> int:
        """pull out the relevant index in the meteo data"""
        return get_astro_time_hour(self.site.utc_time)

    def get_cloud_cover_as_oktas(self, cloud_cover_percentage: int) -> int:
        """convert cloud cover percentage to oktas (eights of sky covered)"""
        import numpy as np
        import math

        if cloud_cover_percentage is None or math.isnan(cloud_cover_percentage):
            raise ValueError("cloud cover percentage is not a number. is open meteo volume up to date?")

        max_oktas = config["meteo"]["max_oktas"]
        percentage_as_oktas = int(np.interp(cloud_cover_percentage, (0, 100), (0, max_oktas)))
        return percentage_as_oktas
