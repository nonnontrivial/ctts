"""Script for getting predicted sky brightness at known locations.

>>> python get_sky_brightness_at_known_sites.py
"""
import subprocess
from pathlib import Path

sites = {
    "home": (43.05148, -78.5767),
    "vera_rubin_lsst": (-30.2446, -70.7494),
    "kitt_peak": (31.9583, -111.5967),
}


if __name__ == "__main__":
    script_path = Path.cwd() / "get_sky_brightness_at_nearest_astro_twilight.py"
    for site_name, (lat, lon) in sites.items():
        cmd = ["python3", script_path, "--lat", str(lat), "--lon", str(lon)]
        subprocess.run(cmd)
