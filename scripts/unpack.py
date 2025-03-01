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
    """extract the contents of the gan zip into ./data/brightness_data"""
    data_dir_path = list(Path.cwd().parent.rglob("brightness_data"))[0]
    assert data_dir_path.exists(), "data dir path does not exist!"
    zip_path = Path.cwd().parent / "gan.zip"
    assert zip_path.exists(), "zip does not exist!"
    log.info(f"unpacking {zip_path} to {data_dir_path}")
    shutil.unpack_archive(zip_path, data_dir_path, "zip")


if __name__ == "__main__":
    main()
