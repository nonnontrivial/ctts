"""Script for getting predicted sky brightness at a location's nearest astronomical twilight.

>>> python get_sky_brightness_at_nearest_astro_twilight.py
"""
import argparse
import logging
import typing as t
from pathlib import Path

import astropy.units as u
import torch
import torch.nn as nn
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.time import Time

logging.basicConfig(
    format="%(asctime)s -> %(levelname)s:%(message)s",
    filename="app.log",
    encoding="utf-8",
    level=logging.DEBUG,
)

features = [
    "Latitude",
    "Longitude",
    "Elevation(m)",
    "CloudCover",
    "UTTimeHour",
    "MoonAlt",
    "MoonAz",
]
HIDDEN_SIZE = 64 * 3
OUTPUT_SIZE = 1
FEATURES_SIZE = len(features)


class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(FEATURES_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE // 2, OUTPUT_SIZE),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


ASTRO_TWILIGHT_DEGS = -18
TEST_LAT = 43.05148
TEST_LON = -78.57732
OPEN_METEO_BASE_URL = "https://api.open-meteo.com"
SITE_NAME = "some-h3-cell"


def get_astro_time_hour(astro_time: Time):
    iso: str = astro_time.iso
    second_segment = iso.split(" ")[1]
    mins_idx = second_segment.index(":")
    return int(second_segment[:mins_idx])


class Site(Observer):
    def __str__(self):
        return f"<astro twilight: {self.utc_astro_twilight.iso}; moon alt: {self.moon_alt}; moon az: {self.moon_az}>"

    @property
    def utc_astro_twilight(self):
        return self.sun_set_time(
            Time.now(), which="nearest", horizon=u.degree * ASTRO_TWILIGHT_DEGS
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
        location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
        observer = Observer(location=location)
        return observer.moon_altaz(self.utc_astro_twilight)


class MeteoClient:
    def __init__(self, site: Site) -> None:
        self.site = site

    def get_response_for_site(self) -> t.Tuple:
        import requests

        lat, lon = self.site.latitude.value, self.site.longitude.value
        res = requests.get(
            f"{OPEN_METEO_BASE_URL}/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,cloud_cover&forecast_days=1"
        )
        res.raise_for_status()
        res_json = res.json()
        idx = self.get_hourly_index_of_astro_twilight()
        cloud_cover = res_json["hourly"]["cloud_cover"][idx]
        cloud_cover = self.get_cloud_cover_as_oktas(cloud_cover)
        return cloud_cover, res_json["elevation"]

    def get_hourly_index_of_astro_twilight(self) -> int:
        return get_astro_time_hour(self.site.utc_astro_twilight)

    def get_cloud_cover_as_oktas(self, cloud_cover_percentage: int):
        import numpy as np

        scaled = np.interp(cloud_cover_percentage, (0, 100), (0, 8))
        return int(scaled)


if __name__ == "__main__":
    import pdb

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lat", required=True, type=str, help="latitude")
    parser.add_argument("--lon", required=True, type=str, help="longitude")
    args = parser.parse_args()
    lat = float(args.lat)
    lon = float(args.lon)
    logging.info(f"registering site at {lat},{lon}")
    location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
    site = Site(location=location, name=SITE_NAME)
    logging.info(f"registered site {site}")
    meteo = MeteoClient(site=site)
    try:
        cloud_cover, elevation = meteo.get_response_for_site()
    except Exception as e:
        logging.error(f"could not get meteo data because {e}")
    else:
        cwd = Path.cwd()
        path_to_state_dict = cwd / "model.pth"
        model = NeuralNetwork()
        model.load_state_dict(torch.load(path_to_state_dict))
        model.eval()

        torch.set_printoptions(sci_mode=False)
        X = torch.tensor(
            [
                site.latitude.value,
                site.longitude.value,
                elevation,
                cloud_cover,
                site.time_hour,
                site.moon_alt,
                site.moon_az,
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            pred = model(X)
            logging.info(f"got prediction {pred} on {X}")
