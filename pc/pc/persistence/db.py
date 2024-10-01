import logging

from pc.config import pg_dsn
from tortoise import Tortoise

log = logging.getLogger(__name__)


async def initialize_db():
    log.info(f"initializing db at {pg_dsn}")
    await Tortoise.init(
        db_url=pg_dsn,
        modules={"models": ["pc.persistence.models"]}
    )
    await Tortoise.generate_schemas()
