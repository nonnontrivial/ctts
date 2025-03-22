import sqlite3
import json
import os
from typing import Any
from pydantic import BaseModel
from fastapi import APIRouter
from ..config import db_path

router = APIRouter()


@router.on_event("startup")
async def startup():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS geojson (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT
    )
    """)
    conn.commit()
    conn.close()


@router.get("/geojson", tags=["geojson"])
async def get_geojson():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM geojson")
    regions = [json.loads(data) for _, data in cursor.fetchall()]
    conn.close()
    return regions


@router.post("/geojson", tags=["geojson"])
async def add_geojson(data: dict[str, Any]):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO geojson (data) VALUES (?)", (json.dumps(data),))
    conn.commit()
    conn.close()
    return {}
