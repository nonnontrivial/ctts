import logging
import os
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import torch
from astropy.coordinates import EarthLocation

from .constants import (
    MODEL_STATE_DICT_FILE_NAME,
    LOGFILE_KEY,
)
from .open_meteo_client import OpenMeteoClient
from .nn import NeuralNetwork
from .observer_site import ObserverSite

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


async def predict_sky_brightness(lat: float, lon: float) -> Prediction:
    """Predict sky brightness at utcnow for the lat and lon"""

    logging.debug(f"registering site at {lat},{lon}")
    location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)

    site = ObserverSite(location=location)
    meteo_client = OpenMeteoClient(site=site)
    try:
        cloud_cover, elevation = await meteo_client.get_values_at_site()

        model = NeuralNetwork()

        path_to_state_dict = Path(__file__).parent / MODEL_STATE_DICT_FILE_NAME
        model.load_state_dict(torch.load(path_to_state_dict))
        model.eval()
    except Exception as e:
        logging.error(f"failed to predict because {e}")
        empty_tensor = torch.empty(4, 4)
        return Prediction(X=empty_tensor, y=empty_tensor)
    else:
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
            predicted_y = model(X)
            return Prediction(X=X, y=predicted_y)
