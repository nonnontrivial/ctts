from fastapi import FastAPI

from .prediction import get_model_prediction_for_nearest_astro_twilight
from .constants import API_PREFIX

app = FastAPI()


@app.get(f"{API_PREFIX}/prediction")
async def get_prediction(
    lat, lon, astro_twilight_type: str = "nearest"
):
    """Get model's sky brightness prediction at the latitude and longitude.

    Optionally pass astro_twilight_type to control which astronomical twilight
    should be used relative to the current time.
    """
    lat, lon = float(lat), float(lon)
    _, y, astro_twilight_iso = await get_model_prediction_for_nearest_astro_twilight(
        lat, lon, astro_twilight_type
    )
    y = round(float(y.item()), 4)
    return {
        "brightness_mpsas": y,
        "astro_twilight": {
            "iso": astro_twilight_iso,
            "type": astro_twilight_type,
        },
    }
