"""Script for getting predicted sky brightness at known sites.

Usually called as part of a cron job workflow, whereby notifications
are generated and sent out.
"""
import csv
import typing as t
import os
from pathlib import Path

from .prediction import get_model_prediction_for_nearest_astro_twilight
from .user import User

notifications_enabled = os.getenv("NOTIFICATIONS_ENABLED", False)

parent_path = Path(__file__).parent
path_to_sites_csv = parent_path / "sites.csv"
path_to_users_csv = parent_path / "users.csv"

with open(path_to_sites_csv, mode="r") as f:
    reader = csv.reader(f)
    sites = list(reader)[1:]


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
        site_id, site_name, lat, lon, *remaining = site
        lat, lon = float(lat), float(lon)
        _, y, astro_twilight = get_model_prediction_for_nearest_astro_twilight(lat, lon)
        y = float(y.item())
        users_watching_site = get_users_watching_site(site_id)
        for user in users_watching_site:
            user_threshold = get_user_threshold_for_site(user, site_id)
            should_notify_user = y >= user_threshold
            print(
                f"user {user} threshold for site {site_id} is {user_threshold}; predicted {y} at {astro_twilight}. notify? {should_notify_user}"
            )
            if should_notify_user and notifications_enabled:
                from .notification import build_user_notification, send_text_to_user

                user_notification = build_user_notification(
                    user, site_name, astro_twilight, y
                )
                sid = send_text_to_user(user_notification)
                print(f"message sent: {sid}")
