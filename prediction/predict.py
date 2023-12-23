"""Script for getting predicted sky brightness at known sites.

Usually called as part of a cron job workflow, whereby notifications
are generated and sent out.
"""
import csv
from pathlib import Path

from .prediction import get_model_prediction_for_nearest_astro_twilight

parent_path = Path(__file__).parent

path_to_csv = parent_path / "sites.csv"
script_path = parent_path / "predict_sky_brightness.py"

with open(path_to_csv, mode="r") as f:
    reader = csv.reader(f)
    sites = list(reader)[1:]

if __name__ == "__main__":
    for site in sites:
        _, lat, lon, *remaining = site
        lat, lon = float(lat), float(lon)
        Xs, y = get_model_prediction_for_nearest_astro_twilight(lat, lon)
        print(Xs, y)
