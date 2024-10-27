import logging
import typing

import asyncpg

from ..config import pg_host,pg_port,pg_user,pg_password,pg_database
from .models import BrightnessObservation

log = logging.getLogger(__name__)
table = "brightness_observation"

async def create_pool() -> typing.Optional[asyncpg.Pool]:
    pool = await asyncpg.create_pool(
        user=pg_user,
        password=pg_password,
        database=pg_database,
        host=pg_host,
        port=pg_port,
        min_size=1,
        max_size=10
    )
    return pool

async def create_brightness_table(pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            uuid UUID PRIMARY KEY,
            lat DOUBLE PRECISION NOT NULL,
            lon DOUBLE PRECISION NOT NULL,
            h3_id TEXT NOT NULL,
            mpsas DOUBLE PRECISION NOT NULL,
            timestamp_utc TIMESTAMPTZ NOT NULL
        );
        """
        )


async def insert_brightness_observation(pool, observation: BrightnessObservation):
    async with pool.acquire() as conn:
        await conn.execute(f"""
        INSERT INTO {table} (uuid, lat, lon, h3_id, mpsas, timestamp_utc)
        VALUES ($1, $2, $3, $4, $5, $6)
        """, observation.uuid, observation.lat, observation.lon, observation.h3_id, observation.mpsas, observation.timestamp_utc)
        log.info(f"Inserted observation: {observation}")
