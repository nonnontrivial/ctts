import typing

from astroplan import Observer
from astropy.table import astropy
from astropy.time import Time
import numpy as np

from .utils import get_astro_time_hour


class ObservationSite(Observer):
    def __init__(self, utc_time: Time, **kwargs):
        super().__init__(**kwargs)

        self.utc_time = utc_time

    @property
    def time_hour(self):
        """mapping of hourly time into sine value"""
        return np.sin(2 * np.pi * get_astro_time_hour(self.utc_time) / 24)

    @property
    def moon_alt(self):
        altaz = self._get_moon_altaz()
        return altaz.alt.value

    @property
    def moon_az(self):
        altaz = self._get_moon_altaz()
        return altaz.az.value

    def _get_moon_altaz(self) -> typing.Any:
        return self.moon_altaz(self.utc_time)
