from typing import Any

import numpy as np
import astropy.units as u
from astroplan import Observer
from astropy.time import Time

from .constants import ASTRO_TWILIGHT_DEGS
from .utils import get_astro_time_hour
from ..model.constants import HOURS_IN_DAY


class Site(Observer):
    def __init__(
        self,
        astro_twilight_type: str,
        location=None,
        timezone="UTC",
        name=None,
        latitude=None,
        longitude=None,
        elevation=0 * u.m,
        pressure=None,
        relative_humidity=None,
        temperature=None,
        description=None,
    ):
        super().__init__(
            location,
            timezone,
            name,
            latitude,
            longitude,
            elevation,
            pressure,
            relative_humidity,
            temperature,
            description,
        )
        self.astro_twilight_type = astro_twilight_type

    def __str__(self):
        return f"<astro twilight: {self.utc_astro_twilight.iso}; >"

    @property
    def utc_astro_twilight(self):
        return self.sun_set_time(
            Time.now(),
            which=self.astro_twilight_type,
            horizon=u.degree * ASTRO_TWILIGHT_DEGS,
        )

    @property
    def hour_sin(self):
        hour_of_astro_twilight = get_astro_time_hour(self.utc_astro_twilight)
        return np.sin(2*np.pi*hour_of_astro_twilight / HOURS_IN_DAY)

    @property
    def hour_cos(self):
        hour_of_astro_twilight = get_astro_time_hour(self.utc_astro_twilight)
        return np.cos(2*np.pi*hour_of_astro_twilight / HOURS_IN_DAY)

    # @property
    # def moon_alt(self):
    #     altaz:Any = self.get_moon_altaz()
    #     return altaz.alt.value

    # @property
    # def moon_az(self):
    #     altaz:Any = self.get_moon_altaz()
    #     return altaz.az.value

    # def get_moon_altaz(self):
    #     return self.moon_altaz(self.utc_astro_twilight)
