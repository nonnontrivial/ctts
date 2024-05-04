from dataclasses import dataclass, asdict
import os

from fastapi import FastAPI, HTTPException, APIRouter

from .pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords
from .prediction.prediction import (
    Prediction,
    predict_sky_brightness,
)

api_version = os.getenv("API_VERSION", "v1")

app = FastAPI()
main_router = APIRouter(prefix=f"/api/{api_version}")


@dataclass
class PredictionResponse:
    """carries sky brightness in mpsas"""
    sky_brightness: float


@main_router.get("/prediction")
async def get_prediction(lat, lon):
    """Predict sky brightness in magnitudes per square arcsecond for a lat and lon"""

    def create_prediction_response(prediction_obj: Prediction) -> PredictionResponse:
        y = round(float(prediction_obj.y.item()), 4)
        return PredictionResponse(sky_brightness=y)

    try:
        lat, lon = float(lat), float(lon)
        prediction = await predict_sky_brightness(lat, lon)
        return asdict(create_prediction_response(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not get prediction because {e}")


@main_router.get("/pollution")
async def get_artificial_light_pollution(lat, lon):
    """Get artificial light pollution at a lat and lon

    Source: https://djlorenz.github.io/astronomy/lp2022/
    """
    try:
        lat, lon = float(lat), float(lon)
        map_image = ArtificialNightSkyBrightnessMapImage()
        pixel_rgba = map_image.get_pixel_value_at_coords(coords=Coords(lat, lon))
        return {channel: pixel_value for channel, pixel_value in zip(("r", "g", "b", "a"), pixel_rgba)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not get light pollution because {e}")


app.include_router(main_router)
