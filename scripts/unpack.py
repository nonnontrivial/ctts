# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from pathlib import Path
import shutil


def main() -> None:
    zip_path = list(Path.cwd().glob("gan-data.zip"))[0]
    shutil.unpack_archive(zip_path, zip_path.parent, "zip")
    print("done")


if __name__ == "__main__":
    main()
