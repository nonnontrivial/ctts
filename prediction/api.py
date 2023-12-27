from fastapi import FastAPI, Request

from .prediction import get_model_prediction_for_nearest_astro_twilight

app = FastAPI()
PREFIX = "api"


@app.get(f"/{PREFIX}/predict")
def predict(lat: str, lon: str, astro_twilight_type: str | None = "nearest"):
    """Predict the sky brightness at the latitude and longitude."""
    lat, lon = float(lat), float(lon)
    _, y, astro_twilight_iso = get_model_prediction_for_nearest_astro_twilight(lat, lon)
    y = round(float(y.item()), 4)
    return {
        "brightness_mpsas": y,
        "astro_twilight": {
            "iso": f"{astro_twilight_iso} UTC",
            "type": astro_twilight_type,
        },
    }
