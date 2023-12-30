from fastapi import FastAPI

from .prediction import get_model_prediction_for_nearest_astro_twilight

app = FastAPI()
PREFIX = "api"


@app.get(f"/{PREFIX}/prediction")
async def get_prediction(
    lat: str, lon: str, astro_twilight_type: str | None = "nearest"
):
    """Get model's sky brightness prediction at the latitude and longitude."""
    lat, lon = float(lat), float(lon)
    _, y, astro_twilight_iso = await get_model_prediction_for_nearest_astro_twilight(
        lat, lon, astro_twilight_type
    )
    y = round(float(y.item()), 4)
    return {
        "brightness_mpsas": y,
        "astro_twilight": {
            "iso": f"{astro_twilight_iso} UTC",
            "type": astro_twilight_type,
        },
    }
