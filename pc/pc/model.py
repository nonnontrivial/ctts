from dataclasses import dataclass


@dataclass
class BrightnessMessage:
    uuid: str
    lat: float
    lon: float
    h3_id: str
    utc_iso: str
    utc_ns: int
    mpsas: float
    model_version: str
