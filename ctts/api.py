import typing as t
import pdb
from fastapi import FastAPI, HTTPException

from .constants import API_PREFIX
from .pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords
from .prediction.prediction import (
    PredictionResponse,
    get_model_prediction_for_astro_twilight_type,
)

app = FastAPI()


def create_prediction_response(prediction: PredictionResponse) -> t.Dict[str, t.Any]:
    y = round(float(prediction.y.item()), 4)
    return {
        "nsb": y,
        "nat": prediction.astro_twilight_iso,
        "model": {},
    }


@app.get(f"{API_PREFIX}/prediction")
async def get_prediction(
    lat, lon, astro_twilight_type: t.Literal["next", "nearest", "previous"]
):
    """Get sky brightness prediction at `lat` and `lon` for an `astro_twilight_type`.

    `astro_twilight_type` is which astronomical twilight should be used relative
    to the current time: next, nearest, or previous.
    """
    try:
        lat, lon = float(lat), float(lon)
        prediction = await get_model_prediction_for_astro_twilight_type(lat, lon, astro_twilight_type)
        return create_prediction_response(prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"error: {e}")


@app.get(f"{API_PREFIX}/pollution")
async def get_artificial_light_pollution(lat, lon):
    """Get artificial light pollution at lat and lon.

    Map data is from 2022: https://djlorenz.github.io/astronomy/lp2022/
    """
    lat, lon = float(lat), float(lon)
    map_image = ArtificialNightSkyBrightnessMapImage()
    pixel_rgba = map_image.get_pixel_values_at_coords(coords=Coords(lat, lon))
    keys = ("r", "g", "b", "a")
    return {k: v for k, v in zip(keys, pixel_rgba)}
