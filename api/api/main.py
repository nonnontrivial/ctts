from dataclasses import asdict
import logging

from fastapi import FastAPI, HTTPException, APIRouter

from .config import api_version, log_level
from .models import PredictionResponse
from .pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords
from .prediction.prediction import (
    Prediction,
    predict_sky_brightness,
)

logging.basicConfig(level=log_level)
app = FastAPI()
main_router = APIRouter(prefix=f"/api/{api_version}")


def create_prediction_response(prediction_obj: Prediction) -> PredictionResponse:
    y = round(float(prediction_obj.y.item()), 4)
    return PredictionResponse(sky_brightness=y)


@main_router.get("/predict")
async def get_prediction(lat, lon):
    """Predict sky brightness in magnitudes per square arcsecond for a lat and lon."""
    try:
        lat, lon = float(lat), float(lon)

        prediction = await predict_sky_brightness(lat, lon)
        return asdict(create_prediction_response(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to predict because {e}")


@main_router.get("/lp")
async def get_artificial_light_pollution(lat, lon):
    """Get artificial light pollution at a lat and lon. Source https://djlorenz.github.io/astronomy/lp2022/"""
    try:
        lat, lon = float(lat), float(lon)

        map_image = ArtificialNightSkyBrightnessMapImage()
        pixel_rgba = map_image.get_pixel_value_at_coords(coords=Coords(lat, lon))

        channels = ("r", "g", "b", "a")
        return {channel: pixel_value for channel, pixel_value in zip(channels, pixel_rgba)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not get light pollution because {e}")


app.include_router(main_router)
