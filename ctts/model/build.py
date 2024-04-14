"""
Script to build (and write to csv) GaN MN data frame.

>>> python -m ctts.model.build
"""

from pathlib import Path
from enum import Enum
from configparser import ConfigParser
import argparse
import sys
import typing as t
import logging

import numpy as np
import httpx
import pandas as pd

from .constants import HOURS_IN_DAY
from .stations import Station, known_stations

from ..pollution.utils import get_luminance_for_color_channels
from ..pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords

gan_mn_dir = Path.cwd() / "data" / "gan_mn"

current_file = Path(__file__)
config = ConfigParser()
config.read(current_file.parent / "config.ini")

gan_mn_dataframe_filename = config.get("file", "gan_mn_dataframe_filename")
log_level = config.getint("log", "level")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
    level=log_level,
)

ansb_map_image = ArtificialNightSkyBrightnessMapImage()

class Features(Enum):
    """
    Columns to be used in feature vector
    """

    HOUR_SIN = "hour_sin"
    HOUR_COS = "hour_cos"
    LAT = "lat"
    LON = "lon"
    ANSB = "ansb"
    CLOUD = "cloud"
    TEMP = "temperature"
    ELEV = "elevation"


class GaNMNData:
    """
    Carries (augmented) dataframe of the Globe at Night Monitoring Network dataset.
    """

    # The columns that we will not be able to build up during prediction request.
    # `temperature` is reconstructed with the result from open meteo.
    nonconstructable_columns = [
        "id",
        "created",
        "received_adjusted",
        "sqmle_serial_number",
        "sensor_frequency",
        "sensor_period_count",
        "sensor_period_second",
        "device_code",
    ]

    def __init__(self, dataset_path: Path) -> None:
        logging.info(f"concatenating {len(list(dataset_path.iterdir()))} files..")

        dfs = [pd.read_csv(f) for f in dataset_path.glob("*.csv")]
        # bring everything into single dataframe
        df = pd.concat(dfs, ignore_index=True)
        # df = df.iloc[:10]
        # pdb.set_trace()

        logging.info("preparing dataset..")
        df = self._sanitize_df(df)

        self.num_rows = df.shape[0]
        logging.info(f"using {self.num_rows} rows with {len(df['device_code'].unique())} unique stations..")

        logging.info("encoding dates..")
        df = self._encode_dates_in_df(df)

        logging.info("applying coordinates..")
        df[Features.LAT.value] = df.apply(self._get_lat_at_row, axis=1)
        df[Features.LON.value] = df.apply(self._get_lon_at_row, axis=1)

        logging.info("applying artificial night sky brightness values..")
        df[Features.ANSB.value] = df.apply(
            self._get_artificial_light_pollution_at_row, axis=1
        )

        logging.info("applying cloud cover..")
        df[Features.CLOUD.value] = df.apply(self._get_cloud_cover_at_row, axis=1)

        logging.info("applying temperature..")
        df[Features.TEMP.value] = df.apply(self._get_temperature_at_row, axis=1)

        logging.info("applying elevation..")
        df[Features.ELEV.value] = df.apply(self._get_elevation_at_row, axis=1)

        logging.info("cleaning up..")
        df = df.drop(columns=self.nonconstructable_columns)

        self.df = df
        self.save_path = dataset_path.parent

    def _sanitize_df(self, gan_frame: pd.DataFrame) -> pd.DataFrame:
        """ensure rows have valid night sky brightness value and are for a station
        with coordinates.

        see "What is the range of the Sky Quality Meters" section of http://www.unihedron.com/projects/darksky/faqsqm.php
        """
        df: t.Any = gan_frame[gan_frame["nsb"] > 7.0]
        df = df[df["device_code"].isin(known_stations)]
        return df.reset_index()

    def _encode_dates_in_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        # see https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
        df = gan_mn_frame
        df["received_utc"] = pd.to_datetime(df["received_utc"])
        df[Features.HOUR_SIN.value] = np.sin(
            2 * np.pi * df["received_utc"].dt.hour / HOURS_IN_DAY
        )
        df[Features.HOUR_COS.value] = np.cos(
            2 * np.pi * df["received_utc"].dt.hour / HOURS_IN_DAY
        )
        return df.reset_index()

    def _get_lat_at_row(self, row: pd.Series):
        device_code = str(row["device_code"])
        return known_stations[device_code][0]

    def _get_lon_at_row(self, row: pd.Series):
        device_code = str(row["device_code"])
        return known_stations[device_code][1]

    def _get_station_at_row(self, row: pd.Series):
        device_code: t.Any = row["device_code"]
        return Station(device_code)

    def _get_cloud_cover_at_row(self, row: pd.Series):
        try:
            station = self._get_station_at_row(row)

            utc: t.Any = row["received_utc"]
            cloud_cover = station.get_cloud_cover(utc)
            logging.info(f"[{row.name}/{self.num_rows}] cloud cover at {station} was {cloud_cover}")
            return cloud_cover
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            logging.error(f"timed out when attempting to get cloud cover: {e}")
            return 0.0
        except Exception as e:
            logging.error(f"could not get cloud cover because {e}")
            return 0.0

    def _get_temperature_at_row(self, row: pd.Series):
        try:
            station = self._get_station_at_row(row)

            utc: t.Any = row["received_utc"]
            temperature = station.get_temperature(utc)
            logging.info(f"[{row.name}/{self.num_rows}] temperature at {station} was {temperature}")
            return temperature
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            logging.error(f"timed out when attempting to get temperature: {e}")
            return 0.0
        except Exception as e:
            logging.error(f"could not get temperature because {e}")
            return 0.0

    def _get_elevation_at_row(self, row: pd.Series):
        try:
            station = self._get_station_at_row(row)
            elevation = station.elevation
            logging.info(f"[{row.name}/{self.num_rows}] elevation at {station} is {elevation}")
            return elevation
        except Exception as e:
            logging.error(f"could not apply elevation because {e}")
            return 0.0

    def _get_artificial_light_pollution_at_row(self, row: pd.Series):
        lat, lon = row[Features.LAT.value], row[Features.LON.value]
        r, g, b, _ = ansb_map_image.get_pixel_values_at_coords(
            Coords(float(lat), float(lon))
        )
        vR, vG, vB = r / 255, g / 255, b / 255
        luminance = get_luminance_for_color_channels(vR, vG, vB)
        return luminance

    @property
    def correlations(self):
        return self.df.corr()

    def write_to_disk(self) -> None:
        self.df.to_csv(self.save_path / gan_mn_dataframe_filename, index=False)


if __name__ == "__main__":
    logging.info(f"running build on csv files within {gan_mn_dir} ..")

    parser = argparse.ArgumentParser(prog="build", description="dataframe writing tool")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    parser.add_argument("--progress", action="store_true", help="show progress")
    args = parser.parse_args()


    try:
        if not gan_mn_dir.exists():
            raise FileNotFoundError(f"{gan_mn_dir} does not exist")

        gan_mn_data = GaNMNData(gan_mn_dir)

        if args.verbose:
            logging.info(f"writing file at {gan_mn_data.save_path / gan_mn_dataframe_filename} ..")
        gan_mn_data.write_to_disk()
        info = gan_mn_data.df.head()
    except ValueError as e:
        logging.error(f"failed to create dataframe because {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"could not build because {e}")
        sys.exit(1)
    else:
        logging.info(f"correlations were:\n{gan_mn_data.correlations}\n\non dataframe\n{info}")
