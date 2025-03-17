import h3
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun

ASTRO_TWILIGHT_DEGS = -18


def get_is_night_across_cells(cell_ids: list[str], time: Time) -> bool:
    coord_list = [h3.cell_to_latlng(cell) for cell in cell_ids]
    night_count = sum(
        get_sun(time)
        .transform_to(
            AltAz(
                obstime=time, location=EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
            )
        )
        .alt.deg
        < ASTRO_TWILIGHT_DEGS
        for lat, lon in coord_list
    )
    return bool(night_count > len(coord_list) / 2)
