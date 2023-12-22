"""Script for getting predicted sky brightness at known locations.

>>> python get_sky_brightness_at_known_sites.py
"""
import csv
import subprocess
from pathlib import Path

parent_path = Path(__file__).parent
path_to_csv = parent_path / "sites.csv"

with open(path_to_csv, mode="r") as f:
    reader = csv.reader(f)
    sites = list(reader)[1:]

if __name__ == "__main__":
    script_path = parent_path / "get_sky_brightness_at_nearest_astro_twilight.py"
    for site in sites:
        lat, lon = site[1], site[2]
        cmd = ["python3", script_path, "--lat", str(lat), "--lon", str(lon)]
        subprocess.run(cmd)
