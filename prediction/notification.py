import os
from dataclasses import dataclass

from twilio.rest import Client

from .user import User

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_NUMBER")

client = Client(account_sid, auth_token)


@dataclass
class Notification(User):
    message: str
    recipient_number: str


def build_user_notification(
    user: User, site_name: str, astro_twilight: str, y: float
) -> Notification:
    message = f"{site_name} is expected to have good seeing ({y:.2f}mpsas) at {astro_twilight}."
    return Notification(**user, message=message, recipient_number="")


def send_text_to_user(notification: Notification) -> str | None:
    message = client.messages.create(
        to=notification.recipient_number, from_=from_number, body=notification.message
    )
    return message.sid
