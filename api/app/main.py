from fastapi import FastAPI
from .routers import inference, geojson

app = FastAPI()
app.include_router(inference.router)
app.include_router(geojson.router)


@app.get("/")
async def root():
    return {}
