# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    zip_path = list(Path.cwd().glob("gan.zip"))[0]
    log.info(f"unpacking {zip_path} to {zip_path.parent}")
    shutil.unpack_archive(zip_path, zip_path.parent, "zip")


if __name__ == "__main__":
    main()
