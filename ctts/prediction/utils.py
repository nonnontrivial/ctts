from datetime import datetime

from astropy.time import Time


def get_astro_time_hour(astro_time: Time) -> int:
    """Gets the hour value from the astro time object."""
    dt = datetime.strptime(str(astro_time.iso), "%Y-%m-%d %H:%M:%S.%f")
    return dt.hour
