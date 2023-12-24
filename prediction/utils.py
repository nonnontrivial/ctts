from astropy.time import Time


def get_astro_time_hour(astro_time: Time) -> int:
    """Get the hour value from the time object."""
    iso: str = astro_time.iso
    second_segment = iso.split(" ")[1]
    mins_idx = second_segment.index(":")
    return int(second_segment[:mins_idx])
