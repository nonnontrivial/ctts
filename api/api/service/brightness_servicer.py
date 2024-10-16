import logging
import uuid
from datetime import timezone, datetime
from pathlib import Path

import astropy.units as u
import torch
from astropy.coordinates import EarthLocation

from ..stubs import brightness_service_pb2, brightness_service_pb2_grpc
from .open_meteo.open_meteo_client import OpenMeteoClient
from .neural_net.nn import NN
from .observer_site import ObserverSite
from . import config

path_to_state_dict = Path(__file__).parent / config["model"]["state_dict_filename"]


class BrightnessServicer(brightness_service_pb2_grpc.BrightnessServiceServicer):
    def GetPollution(self, request, context):
        pass

    def GetBrightnessObservation(self, request, context):
        lat, lon = request.lat, request.lon

        location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
        site = ObserverSite(location=location)
        meteo_client = OpenMeteoClient(site=site)

        try:
            cloud_cover, elevation = meteo_client.get_hourly_values_at_site()
            logging.debug(f"meteo_client response at {lat},{lon} is {cloud_cover}o, {elevation}m")
        except Exception as e:
            import traceback

            logging.error(traceback.format_exc())
            raise ValueError(f"could not get meteo data {e}")
        else:
            model = NN()
            model.load_state_dict(torch.load(path_to_state_dict))
            model.eval()

            torch.set_printoptions(sci_mode=False)
            x = torch.tensor(
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
            logging.debug(f"vector for site is {x}")

            with torch.no_grad():
                predicted_y = model(x)

            time_utc = datetime.now(timezone.utc)
            observation = brightness_service_pb2.BrightnessObservation(
                uuid=str(uuid.uuid4()),
                lat=lat,
                lon=lon,
                utc_iso=time_utc.isoformat(),
                mpsas=predicted_y
            )
            return observation
