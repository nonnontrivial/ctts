"""Script for managing recording api responses with mongodb.

>>> python main.py
"""
import logging
import typing as t
from pathlib import Path

import requests
import typer
from pymongo import MongoClient, collection

DB_NAME = "validation_data"
COLLECTION_NAME = "api_response"
HOST = "localhost"
PORT = 8000

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s -> %(levelname)s: %(message)s"
)

app = typer.Typer()


def get_db(db_name):
    with open(Path.cwd().parent / ".env") as f:
        first_line, *_ = f.read().split("\n")
        idx = first_line.find("=")
        uri = first_line[idx + 1 :]

    client = MongoClient(uri)
    return client[db_name]


def get_collection(db_name: str = DB_NAME, collection_name: str = COLLECTION_NAME):
    """Connect to monogodb instance and get collection back."""
    try:
        logging.info(f"connecting to db {db_name} on collection {collection_name}")
        db = get_db(db_name)
        return db[collection_name]
    except Exception as e:
        logging.error(f"failed to connect: {e}")
        return None


def get_prediction(coordinates: t.Tuple[float]) -> t.Dict:
    lat, lon = coordinates
    url = f"http://{HOST}:{PORT}/api/prediction?lat={lat}&lon={lon}"
    logging.info("getting prediction")
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


@app.command()
def insert(lat: float = -30.2466, lon: float = -70.7494):
    """Get prediction at site and insert into collection."""
    prediction = get_prediction((lat, lon))
    mongo_collection: collection.Collection = get_collection()
    logging.info(f"inserting {prediction} to collection {mongo_collection.name}")
    mongo_collection.insert_one(prediction)
    logging.info("done")


if __name__ == "__main__":
    app()
