"""
Script to untar gzipped data and load it onto disk.

>>> python -m data.unpack
"""

import sys
import tarfile
from pathlib import Path

tar_location = Path(__file__).parent / "gan_mn.tar.gz"
extraction_path = Path(__file__).parent

def cleanup():
    for f in (extraction_path / "gan_mn").glob("._*"):
        f.unlink()

if __name__ == "__main__":
    try:
        if not extraction_path.exists():
            raise FileNotFoundError(f"{extraction_path} does not exist")

        with tarfile.open(tar_location, "r:gz") as tar:
            tar.extractall(extraction_path)

        cleanup()
    except Exception as e:
        print(e)
        sys.exit(1)
    else:
        print("done")
