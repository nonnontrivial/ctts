import pdb
import os
from enum import Enum
import typing as t
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import httpx

from ..prediction.constants import MAX_OKTAS, OPEN_METEO_HISTORICAL_BASE_URL, open_meteo_time_format

# see http://globeatnight-network.org/global-at-night-monitoring-network.html
known_stations = {
    "TAM": (25.0958186, 121.515783),
    "NAOJ": (35.6754121, 139.5343603),
    "HKU": (22.2816752, 114.1370988),
    "NTHU": (24.7961217, 120.9940896),
    "YFAO": (36.8304193, 129.2668557),
    "CNUO": (36.8322677, 127.2949792),
    "LUO": (23.469294, 120.8701045),
    "HKn": (22.383726, 114.1053317),
    "SAAO": (-33.9345359, 18.4745261),
    # "KuO": (),
    "NUM": (47.9230352, 106.9163146),
    "ZSSP": (46.2364857, 17.7291457),
    "Bar": (46.2932137, 17.5891014),
    # "ELO": (),
    "NNO": (32.7590703, 129.8853338),
    "DAO": (36.3995967, 127.3726462),
    "YBAO": (37.1984001, 128.4843463),
    "CHSMO": (36.7394852, 127.2936658),
    "CGAO": (37.0402298, 127.8303813),
    "GSAO": (35.2266359, 127.3665958),
    # "Mac": (),
    # "Tai": (),
    # "Col": (),
    "TST": (22.294288, 114.1693207),
    # "NEO": (),
    "TNO": (18.5738375, 98.4642725),
    # "ISTC": (),
    "CEOU": (37.4593513, 126.9331115),
    "SNUO": (37.4570639, 126.9338594),
    "GYTI": (35.1549191, 129.0935997),
    "AP": (22.3763691, 114.3335007),
    # "iObs": (),
    "UiTM": (3.0697652, 101.4566075),
    # "MFAO": (),
    # "Wux": (),
    "KHG": (22.6297556, 120.3430475),
    "Heh": (24.1621349, 121.2845151),
    # "BhD": (),
    "KP": (22.3115359, 114.1702144),
    "Gdh": (53.2400817, 6.5316701),
    # "Gzc": (),
    # "PAOF": (),
    # "Roo": (),
    # "Sel": (),
    # "Lau": (),
    # "Hor": (),
    # "Lei": (),
    # "Ame": (),
    "SH": (22.2903234, 113.9046078),
    # "Boe": (),
    # "Dez": (),
    # "Loc": (),
    # "Hee": (),
    # "Cap": (),
    "Zwi": (48.99929, 13.2139397),
    "Wad": (52.2108838, 4.5056098),
    # "AHO": ()
    # "Arl": ()
    # "GMARS": ()
    "TSO": (32.6137132, -116.3364648),
    # "TBT": ()
    # "Dal": (),
    # "UO": (),
    "GUT": (54.3716751, 18.6137474),
    "MP": (22.4950569, 114.0159098),
    "FKYC": (22.4881701, 114.1359612),
    # "Oam": (),
    "MBD": (35.9828912, -92.7573844),
    # "LBD": (),
    "Tura": (-35.3208884, 149.0028353),
    # "UBD": (),
    "DNSM": (35.6866333, 128.1769256),
    "Nrnr": (-25, 15.9974197),
    # "BMCO": (),
    # "SkO": (),
    "TSU": (29.8886729, -97.9442456),
    # "Cre": (),
    # "SR": (),
    # "Jub": (),
    # "KO": ()
}

open_meteo_api_protocol = "http"
open_meteo_api_host = os.getenv("OPEN_METEO_API_HOST", "0.0.0.0")
open_meteo_api_port = int(os.getenv("OPEN_METEO_API_PORT", 8080))

class OpenMeteoModels(Enum):
    ERA5_LAND = "era5_land"
    ERA5 = "era5"

class Station:
    open_meteo_historical_base_url = f"{open_meteo_api_protocol}://{open_meteo_api_host}:{open_meteo_api_port}/v1/archive"

    def __init__(self, device_code: str) -> None:
        try:
            lat, lon = known_stations[device_code]
            self.lat = lat
            self.lon = lon
            self.device_code = device_code
        except Exception as e:
           raise ValueError("no known station with device code")

    def __str__(self) -> str:
        return f"Station(device_code={self.device_code})"

    def _scale_cloud_cover(self, cloud_cover: int) -> int:
        return int(np.interp(cloud_cover, (0, 100), (0, MAX_OKTAS)))

    def get_cloud_cover(self, received_utc: pd.Timestamp) -> int:
        one_day: t.Any = pd.Timedelta(days=1)

        start_date = received_utc.strftime(open_meteo_time_format)
        end_date = (received_utc + one_day).strftime(open_meteo_time_format)
        res = httpx.get(self.open_meteo_historical_base_url, params={
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "cloud_cover_mid",
            "models": OpenMeteoModels.ERA5.value
        })
        res.raise_for_status()
        cloud_cover = res.json()["hourly"]["cloud_cover_mid"][received_utc.hour]
        cloud_cover_as_oktas = self._scale_cloud_cover(cloud_cover)
        return cloud_cover_as_oktas


    @property
    def elevation(self) -> float:
        res = httpx.get(self.open_meteo_historical_base_url, params={
            "latitude": self.lat,
            "longitude": self.lon,
            "models": OpenMeteoModels.ERA5_LAND.value
        })
        res.raise_for_status()
        return res.json()["elevation"]
