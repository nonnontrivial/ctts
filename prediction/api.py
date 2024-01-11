import typing as t
from fastapi import FastAPI, HTTPException

from .prediction import Prediction, get_model_prediction_for_astro_twilight_type
from .constants import API_PREFIX

app = FastAPI()
allowed_twilight_types = {"next","previous","nearest"}

def create_response_from_prediction(prediction: Prediction) -> t.Dict:
    y = round(float(prediction.y.item()), 4)
    return {
        "sky_brightness": y,
        "astronomical_twilight_iso": prediction.astro_twilight_iso,
    }

@app.get(f"{API_PREFIX}/prediction")
async def get_prediction(lat, lon, astro_twilight_type: str = "nearest"):
    """Get model's sky brightness prediction at the latitude and longitude.

    Optionally pass query param astro_twilight_type to control which astronomical
    twilight should be used relative to the current time.
    """
    if astro_twilight_type not in allowed_twilight_types:
        raise HTTPException(status_code=400,detail=f"`astro_twilight_type` must be in {allowed_twilight_types}")
    try:
        lat, lon = float(lat), float(lon)
        prediction = await get_model_prediction_for_astro_twilight_type(lat, lon, astro_twilight_type)
        return create_response_from_prediction(prediction)
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"something went wrong: {e}")
