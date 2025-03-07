from fastapi import FastAPI
from .routers import inference

app = FastAPI()
app.include_router(inference.router)


@app.get("/")
async def root():
    return {}
