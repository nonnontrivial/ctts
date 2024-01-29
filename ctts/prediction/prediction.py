import logging
import os
import typing as t
from dataclasses import dataclass
from pathlib import Path

import torch
import astropy.units as u
from astropy.coordinates import EarthLocation

from .constants import  LOGFILE_KEY
from .meteo import MeteoClient
from .site import Site

from ..model.net import LinearNet
from ..model.train_on_dataframe import path_to_state_dict, features, get_device
from ..pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords
from ..pollution.utils import get_luminance_for_color_channels

logfile_name = os.getenv(LOGFILE_KEY)
path_to_logfile = (Path.home() / logfile_name) if logfile_name else None

logging.basicConfig(
    format="%(asctime)s -> %(levelname)s: %(message)s",
    filename=path_to_logfile if bool(path_to_logfile) else None,
    encoding="utf-8",
    level=logging.DEBUG,
)


@dataclass
class PredictionResponse:
    X: torch.Tensor
    y: torch.Tensor
    astro_twilight_iso: str


async def get_model_prediction_for_astro_twilight_type(
    lat: float, lon: float, astro_twilight_type: str
) -> PredictionResponse:
    """Get the sky brightness prediction for an astronomical twilight relative
    to the provided latitude and longitude.
    """
    logging.debug(f"registering site at {lat},{lon}")
    location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
    site = Site(location=location, astro_twilight_type=astro_twilight_type)
    logging.debug(f"registered site {site}")
    try:
        meteo = MeteoClient(site=site)
        temperature, cloud_cover, elevation = await meteo.get_response_for_site()

        ansb_map_image = ArtificialNightSkyBrightnessMapImage()
        r, g, b, _ = ansb_map_image.get_pixel_values_at_coords(Coords(float(lat), float(lon)))
        vR, vG, vB = r/255, g/255, b/255
        luminance = get_luminance_for_color_channels(vR, vG, vB)
    except Exception as e:
        logging.error(f"could not get required data: {e}")
        empty_tensor = torch.empty(1, 1)
        site_astro_twilight_iso = str(site.utc_astro_twilight.iso)
        return PredictionResponse(
            X=empty_tensor, y=empty_tensor, astro_twilight_iso=site_astro_twilight_iso
        )
    else:
        site_astro_twilight_iso = str(site.utc_astro_twilight.iso)
        model = LinearNet(len(features))
        model.load_state_dict(torch.load(path_to_state_dict))
        model.eval()
        torch.set_printoptions(sci_mode=False)
        X = torch.tensor(
            [
                temperature,
                site.hour_sin,
                site.hour_cos,
                site.latitude.value,
                site.longitude.value,
                luminance,
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            output = model(X)
            return PredictionResponse(astro_twilight_iso=site_astro_twilight_iso, X=X, y=output)
