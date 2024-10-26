from pydantic import BaseModel
from datetime import datetime


class BrightnessObservation(BaseModel):
    uuid: str
    lat: float
    lon: float
    h3_id: str
    utc_iso: str
    mpsas: float

class CellCycle(BaseModel):
    start_time_utc: datetime
    end_time_utc: datetime
    duration_s: int
