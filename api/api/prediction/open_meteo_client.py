import logging
import typing as t

from .config import open_meteo_host, open_meteo_port
from .constants import MAX_OKTAS
from .observer_site import ObserverSite
from .utils import get_astro_time_hour


class OpenMeteoClient:
    def __init__(self, site: ObserverSite) -> None:
        self.site = site
        self.url_base = f"http://{open_meteo_host}:{open_meteo_port}"

    async def get_values_at_site(self) -> t.Tuple[int, float]:
        """get cloudcover and elevation values for the observer site"""
        import httpx

        lat, lon = self.site.latitude.value, self.site.longitude.value

        async with httpx.AsyncClient() as client:
            params = {
                "latitude": lat,
                "longitude": lon,
                "models": "ecmwf_ifs04",
                "hourly": "temperature_2m,cloud_cover"
            }
            r = await client.get(f"{self.url_base}/v1/forecast", params=params)
            r.raise_for_status()
            res_json = r.json()

            elevation = float(res_json.get("elevation", 0.))

            idx = self.get_hourly_index_of_site_time()
            cloud_cover = res_json["hourly"]["cloud_cover"][idx]
            cloud_cover = self.get_cloud_cover_as_oktas(cloud_cover)

            return cloud_cover, elevation

    def get_hourly_index_of_site_time(self) -> int:
        """pull out the relevant slice in the meteo data"""
        return get_astro_time_hour(self.site.utc_time)

    def get_cloud_cover_as_oktas(self, cloud_cover_percentage: int):
        """convert percentage to integer oktas value (eights of sky covered)"""
        import numpy as np

        percentage_as_oktas = np.interp(cloud_cover_percentage, (0, 100), (0, MAX_OKTAS))
        return int(percentage_as_oktas)
