from pydantic import BaseModel


class BrightnessObservation(BaseModel):
    uuid: str
    lat: float
    lon: float
    h3_id: str
    utc_iso: str
    mpsas: float
    model_version: str
