import typing
from datetime import datetime
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.table import astropy
from astropy.time import Time
import astropy.units as u
import numpy as np
import requests
from ..config import open_meteo_host
from .open_meteo import OpenMeteo


class Cell(Observer):
    def __init__(self, utc_time: Time, coords: tuple, **kwargs):
        lat, lon = coords
        location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
        super().__init__(location=location, **kwargs)

        self.utc_time = utc_time
        self.open_meteo = OpenMeteo()
        try:
            elevation, cloud_cover = self.open_meteo.get_weather_data(
                lat, lon, self.utc_time
            )
            self._elevation = elevation
            self._cloud_cover = cloud_cover
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")

    @property
    def elevation(self):
        return self._elevation

    @property
    def cloud_cover(self):
        return self._cloud_cover

    @property
    def time_hour(self):
        def get_hour(astro_time: Time) -> int:
            dt = datetime.strptime(str(astro_time.iso), "%Y-%m-%d %H:%M:%S.%f")
            return dt.hour

        return np.sin(2 * np.pi * get_hour(self.utc_time) / 24)

    @property
    def moon_position(self):
        def get_moon_altaz() -> typing.Any:
            return self.moon_altaz(self.utc_time)

        altaz = get_moon_altaz()
        alt = altaz.alt.value
        az = altaz.az.value
        return (alt, az)
