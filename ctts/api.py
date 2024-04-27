import typing as t
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, APIRouter

from .pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords
from .prediction.prediction import (
    Prediction,
    get_model_prediction_for_astro_twilight_type,
)

app = FastAPI()
main_router = APIRouter(prefix="/api/v1")


@dataclass
class PredictionResponse:
    sky_brightness: float


@main_router.get("/prediction")
async def get_prediction(lat, lon):
    """Get sky brightness prediction at `lat` and `lon`
    """

    def create_prediction_response(prediction: Prediction) -> PredictionResponse:
        y = round(float(prediction.y.item()), 4)
        return PredictionResponse(sky_brightness=y)

    try:
        lat, lon = float(lat), float(lon)
        prediction = await get_model_prediction_for_astro_twilight_type(
            lat, lon, "next"
        )
        return asdict(create_prediction_response(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not get prediction because {e}")


@main_router.get("/pollution")
async def get_artificial_light_pollution(lat, lon):
    """Get artificial light pollution at lat and lon (for the year 2022).

    Source: https://djlorenz.github.io/astronomy/lp2022/
    """
    try:
        lat, lon = float(lat), float(lon)
        map_image = ArtificialNightSkyBrightnessMapImage()
        pixel_rgba = map_image.get_pixel_value_at_coords(coords=Coords(lat, lon))
        keys = ("r", "g", "b", "a")
        return {k: v for k, v in zip(keys, pixel_rgba)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not get light pollution because {e}")

app.include_router(main_router)