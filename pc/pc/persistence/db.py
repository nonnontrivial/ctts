import logging
import typing

import asyncpg

from ..config import pg_host, pg_port, pg_user, pg_password, pg_database, brightness_observation_table
from .models import BrightnessObservation, CellCycle

log = logging.getLogger(__name__)

async def create_pg_connection_pool() -> typing.Optional[asyncpg.Pool]:
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

async def setup_table(pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {brightness_observation_table} (
            uuid UUID PRIMARY KEY,
            lat DOUBLE PRECISION NOT NULL,
            lon DOUBLE PRECISION NOT NULL,
            h3_id TEXT NOT NULL,
            mpsas DOUBLE PRECISION NOT NULL,
            timestamp_utc TIMESTAMPTZ NOT NULL
        );
        """
        )


async def insert_brightness_observation(pool: asyncpg.Pool, observation: BrightnessObservation):
    async with pool.acquire() as conn:
        await conn.execute(f"""
        INSERT INTO {brightness_observation_table} (uuid, lat, lon, h3_id, mpsas, timestamp_utc)
        VALUES ($1, $2, $3, $4, $5, $6)
        """, observation.uuid, observation.lat, observation.lon, observation.h3_id, observation.mpsas, observation.timestamp_utc)


async def select_max_brightness_record_in_range(pool: asyncpg.Pool, cycle: CellCycle) -> asyncpg.Record:
    async with pool.acquire() as conn:
        query = f"""
        SELECT *
        FROM {brightness_observation_table}
        WHERE timestamp_utc BETWEEN $1 AND $2
        ORDER BY mpsas DESC
        LIMIT 1;
        """
        record = await conn.fetchrow(query, cycle.start_time_utc, cycle.end_time_utc)
        return record
