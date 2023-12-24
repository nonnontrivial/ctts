"""Script for getting predicted sky brightness at known sites.

Usually called from the `predict.sh` script, as part of a launchd service.
"""
import csv
import logging
import os
import platform
import typing as t
from pathlib import Path

from .message import SiteSummary, build_imessage, send_imessage_to_user
from .prediction import get_model_prediction_for_nearest_astro_twilight
from .site import Site
from .user import User

logfile_name = os.getenv("LOGFILE_NAME", "ctts.log")
path_to_logfile = Path.home() / logfile_name
logging.basicConfig(
    format="%(asctime)s -> %(levelname)s:%(message)s",
    filename=path_to_logfile,
    encoding="utf-8",
    level=logging.INFO,
)


parent_path = Path(__file__).parent
path_to_sites_csv = parent_path / "sites.csv"
path_to_users_csv = parent_path / "users.csv"


def get_sites() -> t.List[Site]:
    with open(path_to_sites_csv, mode="r") as f:
        reader = csv.reader(f)
        _header = next(reader)
        return [Site(*x) for x in reader]


def get_users() -> t.List[User]:
    with open(path_to_users_csv, mode="r") as f:
        reader = csv.reader(f)
        _header = next(reader)
        return [User(*x) for x in reader]


def get_user_by_id(id: str, users: t.List[User]) -> User:
    return next(filter(lambda x: x.id == id, users))


def get_users_watching_site(id: str, users: t.List[User]) -> t.List[User]:
    def get_site_id_in_user_row(user: User):
        return id in user.site_ids.split(":")

    users_watching_site = list(filter(lambda x: get_site_id_in_user_row(x), users))
    return users_watching_site


def get_user_threshold_for_site(user: User, site_id: str) -> float:
    ids = user.site_ids.split(":")
    thresholds = user.site_thresholds.split(":")
    for i in range(len(ids)):
        x = ids[i]
        if x == site_id:
            return float(thresholds[i])


if __name__ == "__main__":
    sites = get_sites()
    users = get_users()
    site_summaries_per_user: t.Dict[str, t.List[SiteSummary]] = {}
    for site in sites:
        lat, lon = float(site.lat), float(site.lon)
        _, y, astro_twilight = get_model_prediction_for_nearest_astro_twilight(lat, lon)
        y = float(y.item())
        users_watching_site = get_users_watching_site(site.id, users)
        for user in users_watching_site:
            user_threshold = get_user_threshold_for_site(user, site.id)
            brightness_meets_threshold = y >= user_threshold
            if brightness_meets_threshold:
                logging.info(
                    f"brightness threshold ({y}/{user_threshold}) met for user {user.id} at site {site.id}"
                )
                summary = SiteSummary(
                    name=site.name, astro_twilight=astro_twilight, predicted_y=y
                )
                if user.id not in site_summaries_per_user:
                    site_summaries_per_user.setdefault(user.id, [summary])
                else:
                    site_summaries_per_user[user.id].append(summary)
    for user_id, site_summaries in site_summaries_per_user.items():
        os_name = platform.system()
        if os_name != "Darwin":
            logging.warning(f"OS must be macOS; detected {os_name}.")
            break
        user = get_user_by_id(user_id, users)
        imessage = build_imessage(user, site_summaries)
        logging.info(
            f"sending message to {user.id} containing {len(site_summaries)} summaries."
        )
        try:
            send_imessage_to_user(imessage)
        except Exception as e:
            logging.error(f"failed to send message to user {user.id}: {e}")
        else:
            logging.info(f"sent message to user {user.id}")
