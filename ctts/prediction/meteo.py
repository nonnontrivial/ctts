import typing as t

from .constants import OPEN_METEO_PRESENT_BASE_URL, MAX_OKTAS
from .site import Site
from .utils import get_astro_time_hour


class MeteoClient:
    def __init__(self, site: Site) -> None:
        self.site = site

    async def get_response_for_site(self) -> t.Tuple[float, int, float]:
        """Gets temperature (celsius), cloud cover (oktas), and elevation (meters)
        for the site.
        """
        import httpx

        lat, lon = self.site.latitude.value, self.site.longitude.value
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{OPEN_METEO_PRESENT_BASE_URL}/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,cloud_cover&forecast_days=1"
            )
            r.raise_for_status()
            res_json = r.json()
            hourly_values = res_json["hourly"]
            idx = self.get_hourly_index_of_astro_twilight()
            temperature = hourly_values["temperature_2m"][idx]
            cloud_cover = hourly_values["cloud_cover"][idx]
            cloud_cover = self.get_cloud_cover_as_oktas(cloud_cover)
            elevation = float(res_json["elevation"])
            return temperature, cloud_cover, elevation

    def get_hourly_index_of_astro_twilight(self) -> int:
        return get_astro_time_hour(self.site.utc_astro_twilight)

    def get_cloud_cover_as_oktas(self, cloud_cover_percentage: int):
        import numpy as np

        percentage_as_oktas = np.interp(cloud_cover_percentage, (0, 100), (0, MAX_OKTAS))
        return int(percentage_as_oktas)
