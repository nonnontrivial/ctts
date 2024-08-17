import logging
import os
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import torch
from astropy.coordinates import EarthLocation

from api.prediction.meteo.open_meteo_client import OpenMeteoClient
from api.prediction.net.nn import NeuralNetwork
from .observer_site import ObserverSite
from .constants import LOGFILE_KEY
from .config import model_state_dict_file_name
from ..config import log_level

logfile_name = os.getenv(LOGFILE_KEY)
path_to_logfile = (Path.home() / logfile_name) if logfile_name else None

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=path_to_logfile if bool(path_to_logfile) else None,
    encoding="utf-8",
    level=log_level,
)


def get_path_to_state_dict():
    return Path(__file__).parent / model_state_dict_file_name


@dataclass
class Prediction:
    X: torch.Tensor
    y: torch.Tensor


path_to_state_dict = get_path_to_state_dict()


async def predict_sky_brightness(lat: float, lon: float) -> Prediction:
    """Predict sky brightness at utcnow for given lat and lon"""

    location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)

    site = ObserverSite(location=location)
    meteo_client = OpenMeteoClient(site=site)

    try:
        cloud_cover, elevation = await meteo_client.get_values_at_site()
        logging.debug(f"meteo_client response at {lat},{lon} is {cloud_cover}o, {elevation}m")
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        raise ValueError(f"meteo data failure: {e}")
    else:
        model = NeuralNetwork()
        logging.debug(f"loading state dict at {path_to_state_dict}")
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

        logging.debug(f"X vector for site is {X}")

        with torch.no_grad():
            predicted_y = model(X)
            return Prediction(X=X, y=predicted_y)
