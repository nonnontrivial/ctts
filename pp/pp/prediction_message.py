from dataclasses import dataclass


@dataclass
class BrightnessMessage:
    uuid: str
    lat: float
    lon: float
    # id of the h3 cell
    h3_id: str
    utc_iso: str
    utc_ns: int
    # magnitudes per square arc second estimated by the model
    mpsas: float
    model_version: str
