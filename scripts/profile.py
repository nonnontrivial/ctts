# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "asyncpg",
# ]
# ///

import asyncio
import logging

import os
import asyncpg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

pg_user = os.getenv("PG_USER", "postgres")
pg_password = os.getenv("PG_PASSWORD", "password")
pg_database = os.getenv("PG_DATABASE", "postgres")
pg_host = os.getenv("PG_HOST", "localhost")
pg_port = int(os.getenv("PG_PORT", 5432))
pg_table = os.getenv("PG_TABLE", "brightness_observation")


async def get_num_brightness_table_rows() -> int:
    conn = await asyncpg.connect(
        user=pg_user,
        password=pg_password,
        database=pg_database,
        host=pg_host,
        port=pg_port,
    )
    count = await conn.fetchval(f"SELECT COUNT(*) FROM {pg_table}")
    await conn.close()
    return count


async def main(period_seconds=5) -> None:
    log.info(f"reading table {pg_table} ..")
    count_start = await get_num_brightness_table_rows()
    await asyncio.sleep(period_seconds)
    count_end = await get_num_brightness_table_rows()
    count_per_second = (count_end - count_start) / period_seconds
    log.info(f"{count_per_second:.2f}/s")


if __name__ == "__main__":
    asyncio.run(main())
