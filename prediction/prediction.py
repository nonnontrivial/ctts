import logging
import os
import typing as t
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import torch
from astropy.coordinates import EarthLocation

from .constants import (
    MODEL_STATE_DICT_FILE_NAME,
    LOGFILE_KEY,
    SITE_NAME,
)
from .meteo import MeteoClient
from .nn import NeuralNetwork
from .site import Site

logfile_name = os.getenv(LOGFILE_KEY)
path_to_logfile = (Path.home() / logfile_name) if logfile_name else None
logging.basicConfig(
    format="%(asctime)s -> %(levelname)s: %(message)s",
    filename=path_to_logfile if bool(path_to_logfile) else None,
    encoding="utf-8",
    level=logging.DEBUG,
)

@dataclass
class Prediction:
    X: torch.Tensor
    y: torch.Tensor
    astro_twilight_iso: str

async def get_model_prediction_for_astro_twilight_type(
    lat: float, lon: float, astro_twilight_type: str
) -> Prediction:
    """Get the sky brightness prediction for an astronomical twilight relative
    to the provided latitude and longitude.
    """
    logging.debug(f"registering site at {lat},{lon}")
    location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
    site = Site(
        location=location, name=SITE_NAME, astro_twilight_type=astro_twilight_type
    )
    site_astro_twilight_iso = str(site.utc_astro_twilight.iso)
    logging.debug(f"registered site {site}")
    meteo = MeteoClient(site=site)
    try:
        cloud_cover, elevation = await meteo.get_response_for_site()
    except Exception as e:
        logging.error(f"could not get meteo data because {e}")
        empty = torch.empty(4,4)
        return Prediction(X=empty,y=empty,astro_twilight_iso=site_astro_twilight_iso)
    else:
        path_to_state_dict = Path(__file__).parent / MODEL_STATE_DICT_FILE_NAME
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
            return Prediction(astro_twilight_iso=site_astro_twilight_iso,X=X,y=pred)
