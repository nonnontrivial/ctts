import astropy.units as u
from astroplan import Observer
from astropy.time import Time

from .constants import ASTRO_TWILIGHT_DEGS
from .utils import get_astro_time_hour


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
        return f"<astro twilight: {self.utc_astro_twilight.iso}; moon alt: {self.moon_alt}; moon az: {self.moon_az}>"

    @property
    def utc_astro_twilight(self):
        return self.sun_set_time(
            Time.now(),
            which=self.astro_twilight_type,
            horizon=u.degree * ASTRO_TWILIGHT_DEGS,
        )

    @property
    def time_hour(self):
        import numpy as np

        return np.sin(2 * np.pi * get_astro_time_hour(self.utc_astro_twilight) / 24)

    @property
    def moon_alt(self):
        altaz = self.get_moon_altaz()
        return altaz.alt.value

    @property
    def moon_az(self):
        altaz = self.get_moon_altaz()
        return altaz.az.value

    def get_moon_altaz(self):
        return self.moon_altaz(self.utc_astro_twilight)
