"""Script for getting predicted sky brightness at known locations.

>>> python get_sky_brightness_at_known_sites.py
"""
import subprocess
from pathlib import Path

from google.cloud import sql

script_path = Path.cwd() / "get_sky_brightness_at_nearest_astro_twilight.py"
sites = {
    "home": (43.05148, -78.5767),
    "vera_rubin_lsst": (-30.2446, -70.7494),
    "kitt_peak": (31.9583, -111.5967),
}


def main():
    """Get predictions for all sites known about ahead of time."""
    for site_name, (lat, lon) in sites.items():
        print(f"> processing site {site_name}")
        cmd = ["python3", script_path, "--lat", str(lat), "--lon", str(lon)]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
