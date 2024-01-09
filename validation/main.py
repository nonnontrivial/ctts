"""Script for storing api responses in mongodb collection.

To get a prediction and add it to a collection:
>>> python main.py insert <lat> <lon> <collection_name>

To get all items from a mongo collection and write them to disk as a .csv:
>>> python main.py transcribe <collection_name> <relative_dest_path>

To wipe all items from a mongo collection.
>>> python main.py wipe <collection_name>
"""
import logging
import csv
import typing as t
from pathlib import Path

import requests
import typer
from pymongo import MongoClient, collection

# from ..prediction.constants import API_PREFIX

API_PREFIX="/api/v1"
DB_NAME = "validation_data"
COLLECTION_NAME = "api_response"

DEFAULT_LAT = -30.2466
DEFAULT_LON = -70.7494

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


def get_collection(collection_name: str):
    """Connect to monogodb instance and get collection back."""
    logging.info(f"connecting to db {DB_NAME} on collection {collection_name}")
    db = get_db(DB_NAME)
    return db[collection_name]


def get_prediction(coordinates: t.Tuple[float, float]) -> t.Dict:
    lat, lon = coordinates
    url = f"http://{HOST}:{PORT}{API_PREFIX}/prediction?lat={lat}&lon={lon}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


@app.command()
def insert(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, collection_name = COLLECTION_NAME):
    """Get prediction at site and insert into collection."""
    try:
        prediction = get_prediction((lat, lon))
        mongo_collection: collection.Collection = get_collection(collection_name)
    except Exception as e:
        logging.error(f"!{e}")
    else:
        logging.info(f"inserting {prediction} to collection {mongo_collection.name}")
        mongo_collection.insert_one(prediction)
        logging.info("done")

@app.command()
def transcribe(collection_name: str, csv_destination: str):
    """Get all items in collection and write to csv."""
    try:
        mongo_collection: collection.Collection = get_collection(collection_name)
        items_in_collection = list(mongo_collection.find())
    except Exception as e:
        logging.error(f"failed to query collection {collection_name}; {e}")
    else:
        logging.info(f"found {len(items_in_collection)} items in collection {collection_name}")

        BRIGHTNESS_KEY="brightness_mpsas"
        items_in_collection = [{"y":x[BRIGHTNESS_KEY],"timestamp":x["astro_twilight"]["iso"]} for x in items_in_collection]
        headers = list(items_in_collection[0].keys())

        csv_destination_path = Path(csv_destination)
        with open(csv_destination_path,"w",newline="") as file:
            writer = csv.DictWriter(file,fieldnames=headers)
            logging.info(f"writing to {csv_destination_path.resolve()}..")
            writer.writeheader()
            writer.writerows(items_in_collection)
            logging.info("done")

@app.command()
def wipe(collection_name:str):
    """Remove all items from a collection."""
    try:
        mongo_collection: collection.Collection = get_collection(collection_name)
        items_in_collection = list(mongo_collection.find({}))
        logging.info(f"removing {len(items_in_collection)} items")
        mongo_collection.delete_many({})
    except Exception as e:
        logging.error(f"failed to remove items from collection {collection_name}; {e}")
    else:
        items_in_collection = list(mongo_collection.find({}))
        logging.info(f"collection {collection_name} now contains {len(items_in_collection)} items")

if __name__ == "__main__":
    app()
