from astroplan import Observer
from astropy.time import Time

from .utils import get_astro_time_hour


class ObserverSite(Observer):
    """A location on the earth, loaded with data around moon altitude and azimuth at the utc time"""

    def __str__(self):
        return f"<astro twilight: {self.utc_time.iso}; moon alt: {self.moon_alt}; moon az: {self.moon_az}>"

    @property
    def utc_time(self):
        return Time.now()

    @property
    def time_hour(self):
        """mapping of hourly time into sine value"""
        import numpy as np

        return np.sin(2 * np.pi * get_astro_time_hour(self.utc_time) / 24)

    @property
    def moon_alt(self):
        altaz = self.get_moon_altaz()
        return altaz.alt.value

    @property
    def moon_az(self):
        altaz = self.get_moon_altaz()
        return altaz.az.value

    def get_moon_altaz(self):
        return self.moon_altaz(self.utc_time)
