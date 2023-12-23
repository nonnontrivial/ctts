"""Script for getting predicted sky brightness at known sites.

>>> python -m prediction
"""
import csv
import subprocess
from pathlib import Path

parent_path = Path(__file__).parent

path_to_csv = parent_path / "sites.csv"
script_path = parent_path / "predict_sky_brightness.py"

with open(path_to_csv, mode="r") as f:
    reader = csv.reader(f)
    sites = list(reader)[1:]

if __name__ == "__main__":
    for site in sites:
        _, lat, lon, *remaining = site
        cmd = ["python3", script_path, "--lat", str(lat), "--lon", str(lon)]
        subprocess.run(cmd)
