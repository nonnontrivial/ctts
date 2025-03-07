import requests
import numpy as np
from astropy.time import Time
from ..config import open_meteo_host


class OpenMeteo:
    forecast_url = f"http://{open_meteo_host}:8080/v1/forecast"
    elevation_url = f"http://{open_meteo_host}:8080/v1/elevation"
    model = "gfs_global"

    def get_weather_data(self, lat: float, lon: float, time: Time) -> tuple[float, int]:
        def get_cloud_cover_as_oktas(percentage: int):
            percentage_as_oktas = int(np.interp(percentage, (0, 100), (0, 8)))
            return percentage_as_oktas

        res = requests.get(
            self.forecast_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "models": self.model,
                "current": "temperature_2m,cloud_cover",
                "timeformat": "iso8601",
                "timezone": "UTC",
            },
        )
        res.raise_for_status()
        data = res.json()
        elevation = data.get("elevation", 0.0)
        cloud_cover = get_cloud_cover_as_oktas(
            int(data.get("current", {}).get("cloud_cover", 0) or 0)
        )
        return elevation, cloud_cover
