"""Script for getting predicted sky brightness at known sites.

Usually called as part of a cron job workflow, whereby notifications
are generated and sent out.
"""
import csv
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

from .prediction import get_model_prediction_for_nearest_astro_twilight

parent_path = Path(__file__).parent
path_to_sites_csv = parent_path / "sites.csv"
path_to_users_csv = parent_path / "users.csv"

with open(path_to_sites_csv, mode="r") as f:
    reader = csv.reader(f)
    sites = list(reader)[1:]


@dataclass
class User:
    id: str
    site_ids: t.List[str]
    site_thresholds: t.Dict[str, float]


users = []
with open(path_to_users_csv, mode="r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        user = User(*row)
        users.append(user)


def get_users_watching_site(id: str) -> t.List[User]:
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
    for site in sites:
        site_id, name, lat, lon, *remaining = site
        lat, lon = float(lat), float(lon)
        _, y = get_model_prediction_for_nearest_astro_twilight(lat, lon)
        y = float(y.item())
        users_watching_site = get_users_watching_site(site_id)
        for user in users_watching_site:
            user_threshold = get_user_threshold_for_site(user, site_id)
            should_notify_user = y >= user_threshold
            print(
                f"user {user} threshold for site {site_id} is {user_threshold}; predicted {y}. notify? {should_notify_user}"
            )
