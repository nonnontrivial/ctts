from dataclasses import dataclass


@dataclass
class BrightnessMessage:
    uuid: str
    lat: float
    lon: float
    # id of the h3 cell
    h3_id: str
    # utc datetime that this message was published
    utc: str
    # magnitudes per square arc second estimated by the model
    mpsas: float
    model_version: str
