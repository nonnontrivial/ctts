import typing as t
from fastapi import FastAPI, HTTPException

from .prediction.prediction import Prediction, get_model_prediction_for_astro_twilight_type
from .constants import API_PREFIX

app = FastAPI()

def create_response_from_prediction(prediction: Prediction) -> t.Dict:
    y = round(float(prediction.y.item()), 4)
    return {
        "sky_brightness": y,
        "astronomical_twilight_iso": prediction.astro_twilight_iso,
    }

@app.get(f"{API_PREFIX}/prediction")
async def get_prediction(lat, lon, astro_twilight_type: t.Literal["next","nearest","previous"]):
    """Get sky brightness prediction at lat and lon for an astro_twilight_type.

    `astro_twilight_type` is which astronomical twilight should be used relative
    to the current time: next, nearest, or previous.
    """
    try:
        lat, lon = float(lat), float(lon)
        prediction = await get_model_prediction_for_astro_twilight_type(lat, lon, astro_twilight_type)
        return create_response_from_prediction(prediction)
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"something went wrong: {e}")

@app.get(f"{API_PREFIX}/pollution")
async def get_artificial_light_pollution(lat, lon):
    """Get artificial light pollution at a lat and lon."""
    pass
